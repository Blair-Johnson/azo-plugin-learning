"""Learning add-on: a persistent model of what the user understands.

Working agents record passive observations with
``characterize_user_understanding``. Those observations land in a global queue.

``/study`` turns the current session into a study session. Study sessions see
a different tool set: they build a knowledge graph of true statements linked
by dependencies, and record evidence of understanding against its nodes.
``next_study_step`` runs the session as a repeating cycle of review, probing,
and teaching, so the user only has to answer and learn.

The graph is the only learning data structure. Nothing stores a schedule or a
mastery score: each node's review timing, hole and edge scores are derived from
its evidence and its neighbours' evidence whenever they are read. Review is a
ranking, so it can never build up a backlog of overdue items.

Everything lives in one SQLite database under the Agent Zoo state home, shared
by every project and session. Thresholds come from the plugin's YAML config.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from agent_utils import Feature
from agent_utils.components import (
    ExcludeForgotten,
    MessageRenderer,
    RenderTransformRegistrar,
)
from agent_utils.tool import Tool
from agent_utils.types import MODEL_RENDER_CHANNEL
from agent_zoo.projects import resolve_state_root
from agent_zoo.runtime.commands import Command, CommandResult

_PLUGIN_NAME = "azo-plugin-learning"
STUDY_FLAG = "learning_study_active"
CYCLE = "learning_study_cycle"
QUESTION_PHASES = ("review", "probe", "teach")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, dict[str, float | int]] = {
    "cycle": {
        # Review questions at the start of each cycle, when nodes are due.
        "review_questions": 2,
        # Probe questions before teaching, if the user doesn't miss first.
        "probe_questions": 3,
        # Times a node is taught before moving on and leaving it to review.
        "max_teach_attempts": 2,
        # How far from the focus node probing may wander, in dependency links.
        "focus_hops": 2,
        # A gap this long since the last step earns a one-line recap.
        "recap_after_minutes": 20,
    },
    "accuracy": {
        # Recent answers used to judge whether questions are too hard or easy.
        "window_answers": 10,
        "window_hours": 12,
        "min_answers": 4,
        # Below low: one probe per cycle and easier targets. Above high: harder.
        "low": 0.75,
        "high": 0.90,
    },
    "review": {
        # Review interval after a lapse or a first success.
        "base_interval_days": 1.0,
        # Each further spaced success multiplies the interval by this.
        "growth_factor": 3.0,
        # Successes closer together than this count as one.
        "min_success_gap_hours": 8.0,
        # A node is due once time since its last evidence reaches this
        # multiple of its interval.
        "due_ratio": 1.0,
        # Extra review priority per node that depends on this one.
        "dependent_weight": 0.25,
    },
    "scoring": {
        # Evidence weight halves over this many days, for hole/edge scores.
        "half_life_days": 30.0,
        # Weight of a passive observation relative to a study answer.
        "observation_weight": 0.4,
        # How much neighbours' evidence counts toward a node's scores.
        "neighbour_weight": 0.5,
    },
}


def config_path() -> Path:
    return resolve_state_root() / "plugin-configs" / _PLUGIN_NAME / "config" / "learning.yaml"


def load_config() -> dict[str, Any]:
    """Return defaults overlaid with the user's config file.

    Read on every call so edits apply without a reload. Invalid values keep
    their defaults and are reported in ``_warnings``.
    """

    cfg: dict[str, Any] = {name: dict(values) for name, values in DEFAULT_CONFIG.items()}
    warnings: list[str] = []
    cfg["_warnings"] = warnings
    path = config_path()
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError:
        return cfg
    except (OSError, yaml.YAMLError) as exc:
        warnings.append(f"could not read {path}: {exc}")
        return cfg
    if not isinstance(raw, dict):
        warnings.append(f"{path} is not a mapping")
        return cfg
    for section, defaults in DEFAULT_CONFIG.items():
        values = raw.get(section) or {}
        if not isinstance(values, dict):
            warnings.append(f"{section} is not a mapping")
            continue
        for key, value in values.items():
            if key not in defaults:
                warnings.append(f"unknown setting {section}.{key}")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                warnings.append(f"{section}.{key} must be a non-negative number")
                continue
            cfg[section][key] = type(defaults[key])(value)
    return cfg


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    topic TEXT NOT NULL,
    evidence TEXT NOT NULL,
    understands INTEGER NOT NULL CHECK (understands IN (0, 1)),
    project TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    user_entry_index INTEGER
);
CREATE INDEX IF NOT EXISTS observations_created_at ON observations (created_at);

CREATE TABLE IF NOT EXISTS observation_dismissals (
    observation_id INTEGER PRIMARY KEY REFERENCES observations (id),
    dismissed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    statement TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
    USING fts5 (statement, tokenize = 'porter unicode61');

CREATE TABLE IF NOT EXISTS dependencies (
    node_id INTEGER NOT NULL REFERENCES nodes (id),
    depends_on_id INTEGER NOT NULL REFERENCES nodes (id),
    created_at TEXT NOT NULL,
    PRIMARY KEY (node_id, depends_on_id)
);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id INTEGER NOT NULL REFERENCES nodes (id),
    understands INTEGER NOT NULL CHECK (understands IN (0, 1)),
    evidence TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('study', 'observation')),
    observation_id INTEGER REFERENCES observations (id),
    project TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    user_entry_index INTEGER,
    -- Set when this row is the graded answer to a study-step question.
    question_phase TEXT
);
CREATE INDEX IF NOT EXISTS evidence_node ON evidence (node_id);
CREATE UNIQUE INDEX IF NOT EXISTS evidence_observation_node
    ON evidence (observation_id, node_id) WHERE observation_id IS NOT NULL;
"""


def db_path() -> Path:
    return resolve_state_root() / "learning" / "learning.sqlite"


def _migrate(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(evidence)")}
    if "question_phase" not in columns:
        try:
            conn.execute("ALTER TABLE evidence ADD COLUMN question_phase TEXT")
        except sqlite3.OperationalError as exc:
            # Another session may have migrated concurrently.
            if "duplicate column" not in str(exc):
                raise
    conn.execute(
        "CREATE INDEX IF NOT EXISTS evidence_answers ON evidence (observed_at)"
        " WHERE question_phase IS NOT NULL"
    )


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = db_path() if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    # Many concurrent agent sessions write to the same database.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(moment: datetime | None = None) -> str:
    return (moment or _now()).isoformat(timespec="seconds")


def _parse_time(value: str) -> datetime | None:
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def _run(work):
    conn = connect()
    try:
        with conn:
            return work(conn)
    finally:
        conn.close()


_PENDING_OBSERVATIONS = (
    "SELECT * FROM observations o WHERE NOT EXISTS"
    " (SELECT 1 FROM evidence e WHERE e.observation_id = o.id) AND NOT EXISTS"
    " (SELECT 1 FROM observation_dismissals d WHERE d.observation_id = o.id)"
    " ORDER BY created_at, id"
)


# ---------------------------------------------------------------------------
# Transcript pointers
# ---------------------------------------------------------------------------

def _current_session_id(state: Any) -> str:
    run_db = getattr(state, "run_db", None)
    sid = str(getattr(run_db, "session_id", "") or "").strip()
    if sid:
        return sid
    return str(getattr(state, "_session_id", "") or "").strip()


def _current_project(state: Any) -> str:
    for attr in ("_project_name", "project_name"):
        value = str(getattr(state, attr, "") or "").strip()
        if value:
            return value
    run_db = getattr(state, "run_db", None)
    agent_db = getattr(run_db, "agent_db", None)
    db = str(getattr(agent_db, "path", "") or "").strip()
    if db:
        path = Path(db).expanduser()
        if path.name == "db.sqlite" and path.parent.name:
            return path.parent.name
    return ""


def _latest_user_entry_index(state: Any) -> int | None:
    for entry in reversed(getattr(state, "entries", None) or []):
        if getattr(entry, "system_generated", False):
            continue
        if getattr(entry, "role", None) == "user":
            return getattr(entry, "index", None)
    return None


def _pointer(state: Any) -> tuple[str, str, int | None]:
    return (
        _current_project(state),
        _current_session_id(state),
        _latest_user_entry_index(state),
    )


def _format_pointer(row: sqlite3.Row) -> str:
    if not row["session_id"]:
        return "no transcript pointer"
    entry = row["user_entry_index"]
    where = f"entry {entry}" if entry is not None else "unknown entry"
    return f"project {row['project'] or '?'}, session {row['session_id']}, {where}"


# ---------------------------------------------------------------------------
# Derived node state
# ---------------------------------------------------------------------------

@dataclass
class NodeState:
    id: int
    statement: str
    # Recency-weighted sum of +1 (understands) and -1 (doesn't) evidence.
    net: float = 0.0
    # Recency-weighted amount of direct evidence, regardless of direction.
    weight: float = 0.0
    count: int = 0
    last_seen: str = ""
    depends_on: tuple[int, ...] = ()
    dependents: tuple[int, ...] = ()
    # Positive evidence on nodes that depend on this one.
    support_above: float = 0.0
    # Positive and negative evidence on nodes this one depends on.
    known_below: float = 0.0
    doubt_below: float = 0.0
    # High when what the user knows rests on this node, but its own evidence
    # is thin, old, or negative.
    hole_score: float = 0.0
    # High when the node failed or is untested while its foundations look solid.
    edge_score: float = 0.0
    # Review timing, derived from the times of the node's evidence.
    last_evidence: datetime | None = None
    spaced_successes: int = 0
    interval_days: float = 0.0
    # Time since the last evidence, as a multiple of the interval.
    urgency: float = 0.0
    review_priority: float = 0.0


def _decay(moment: datetime, now: datetime, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    age_days = max((now - moment).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (age_days / half_life_days)


def _derive_review(state: NodeState, history, now: datetime, review: dict) -> None:
    """Set review timing from the node's evidence, oldest first.

    Each success at least ``min_success_gap_hours`` after the last counted
    success extends the interval; any miss resets it.
    """

    if not history:
        return
    gap = timedelta(hours=review["min_success_gap_hours"])
    successes, last_success = 0, None
    for moment, understands in history:
        if understands:
            if last_success is None or moment - last_success >= gap:
                successes += 1
                last_success = moment
        else:
            successes, last_success = 0, None
    state.spaced_successes = successes
    state.last_evidence = history[-1][0]
    state.interval_days = review["base_interval_days"] * review["growth_factor"] ** max(
        successes - 1, 0
    )
    elapsed_days = max((now - state.last_evidence).total_seconds() / 86400.0, 0.0)
    state.urgency = elapsed_days / max(state.interval_days, 1.0 / 1440)
    state.review_priority = state.urgency * (
        1.0 + review["dependent_weight"] * len(state.dependents)
    )


def node_states(
    conn: sqlite3.Connection,
    now: datetime | None = None,
    cfg: dict | None = None,
) -> dict[int, NodeState]:
    now = now or _now()
    cfg = cfg or load_config()
    scoring, review = cfg["scoring"], cfg["review"]
    source_weights = {"study": 1.0, "observation": scoring["observation_weight"]}

    states = {
        row["id"]: NodeState(row["id"], row["statement"])
        for row in conn.execute("SELECT id, statement FROM nodes")
    }
    history: dict[int, list[tuple[datetime, bool]]] = {node: [] for node in states}
    for row in conn.execute(
        "SELECT node_id, understands, observed_at, source_kind FROM evidence"
        " ORDER BY observed_at, id"
    ):
        state = states.get(row["node_id"])
        moment = _parse_time(row["observed_at"])
        if state is None or moment is None:
            continue
        weight = source_weights.get(row["source_kind"], 0.0) * _decay(
            moment, now, scoring["half_life_days"]
        )
        state.net += weight if row["understands"] else -weight
        state.weight += weight
        state.count += 1
        state.last_seen = max(state.last_seen, row["observed_at"])
        history[state.id].append((moment, bool(row["understands"])))

    depends_on: dict[int, list[int]] = {node: [] for node in states}
    dependents: dict[int, list[int]] = {node: [] for node in states}
    for row in conn.execute("SELECT node_id, depends_on_id FROM dependencies"):
        if row["node_id"] in states and row["depends_on_id"] in states:
            depends_on[row["node_id"]].append(row["depends_on_id"])
            dependents[row["depends_on_id"]].append(row["node_id"])
    for node, state in states.items():
        state.depends_on = tuple(sorted(depends_on[node]))
        state.dependents = tuple(sorted(dependents[node]))

    neighbour = scoring["neighbour_weight"]
    for node, state in states.items():
        state.support_above = sum(max(states[d].net, 0.0) for d in state.dependents)
        state.known_below = sum(max(states[p].net, 0.0) for p in state.depends_on)
        state.doubt_below = sum(max(-states[p].net, 0.0) for p in state.depends_on)
        state.hole_score = (
            state.support_above
            * neighbour
            * (1.0 / (1.0 + state.weight) + max(-state.net, 0.0))
        )
        if state.net <= 0:
            state.edge_score = (
                max(-state.net, 0.0) + state.known_below * neighbour
            ) / (1.0 + state.doubt_below)
        _derive_review(state, history[node], now, review)
    return states


def _describe(state: NodeState) -> str:
    if state.count:
        evidence = (
            f"{state.count} evidence record(s), last {state.last_seen[:10]}, "
            f"net {state.net:+.2f}"
        )
    else:
        evidence = "no direct evidence"
    return f"node {state.id}: {state.statement} ({evidence})"


def _describe_review(state: NodeState, now: datetime) -> str:
    if state.last_evidence is None:
        return "Never reviewed."
    days = (now - state.last_evidence).total_seconds() / 86400.0
    return (
        f"Last evidence {days:.1f} days ago; review interval {state.interval_days:g} days "
        f"after {state.spaced_successes} spaced success(es)."
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class LearningTool(Tool):
    """A tool shown only inside, or only outside, study sessions."""

    def __init__(self, fn, *, study: bool, **kwargs):
        super().__init__(fn, **kwargs)
        self.study = study

    def available(self, state) -> bool:
        return bool(getattr(state, STUDY_FLAG, False)) == self.study


_STATE_READS = {"entries", "_session_id", "run_db", "_project_name", STUDY_FLAG, CYCLE}


def _learning_tool(*, study: bool, writes: set | None = None, init: dict | None = None):
    def wrap(fn):
        return LearningTool(
            fn, study=study, optional_reads=set(_STATE_READS), writes=writes, init=init
        )

    return wrap


def _require_text(**fields: str) -> tuple[dict[str, str], str | None]:
    cleaned = {}
    for name, value in fields.items():
        text = str(value or "").strip()
        if not text:
            return {}, f"ERROR: {name} must be non-empty."
        cleaned[name] = text
    return cleaned, None


def _missing_nodes(conn: sqlite3.Connection, ids) -> list[int]:
    return [
        node for node in ids
        if conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node,)).fetchone() is None
    ]


def _depends_transitively(conn: sqlite3.Connection, start: int, target: int) -> bool:
    row = conn.execute(
        """
        WITH RECURSIVE reach(id) AS (
            SELECT ?
            UNION
            SELECT d.depends_on_id FROM dependencies d JOIN reach r ON d.node_id = r.id
        )
        SELECT 1 FROM reach WHERE id = ?
        """,
        (start, target),
    ).fetchone()
    return row is not None


@_learning_tool(study=False)
def characterize_user_understanding(
    topic: str,
    evidence: str,
    understands: bool,
    state=None,
) -> str:
    """Characterize whether the user understands something, with brief evidence.

    Call this whenever the user reveals what they do or don't understand,
    even if it has nothing to do with your current task. That includes asking
    what something means, asking for an explanation, seeming unsure, saying
    something incorrect, explaining a concept correctly without help, or
    catching a mistake you made. It returns immediately and does not affect
    your work; you do not need to mention it to the user.

    Args:
        topic: A specific description of what the user does or doesn't
            understand, as precise as the evidence allows. For example,
            "why the GIL stops CPU-bound Python threads from running in
            parallel, while I/O-bound threads still benefit", not "Python GIL".
        evidence: What the user said or did that revealed knowledge or lack
            thereof on the topic.
        understands: True if the user understands the topic, False if not.
    """

    fields, error = _require_text(topic=topic, evidence=evidence)
    if error:
        return error
    if not isinstance(understands, bool):
        return "ERROR: understands must be true or false."

    row = (_timestamp(), fields["topic"], fields["evidence"], int(understands), *_pointer(state))
    _run(lambda conn: conn.execute(
        "INSERT INTO observations (created_at, topic, evidence, understands,"
        " project, session_id, user_entry_index) VALUES (?, ?, ?, ?, ?, ?, ?)",
        row,
    ))
    return "Recorded."


@_learning_tool(study=True)
def search_knowledge(query: str, limit: int = 10, state=None) -> str:
    """Search the user's knowledge graph for nodes matching a description.

    Always search before adding a node, and reuse an existing node that
    states the same idea in other words.

    Args:
        query: Words describing the idea to look for.
        limit: Maximum number of nodes to return.
    """

    terms = re.findall(r"\w+", str(query or ""))
    if not terms:
        return "ERROR: query must contain words to search for."
    match = " OR ".join('"' + term + '"' for term in terms)
    limit = max(1, min(int(limit), 50))

    def work(conn):
        ids = [
            row["rowid"]
            for row in conn.execute(
                "SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH ?"
                " ORDER BY bm25(nodes_fts) LIMIT ?",
                (match, limit),
            )
        ]
        states = node_states(conn)
        return [states[node] for node in ids if node in states]

    found = _run(work)
    if not found:
        return "No matching nodes."
    return "\n".join(_describe(state) for state in found)


@_learning_tool(study=True)
def add_knowledge(statement: str, depends_on: list[int] | None = None, state=None) -> str:
    """Add a node to the user's knowledge graph.

    Search with search_knowledge first. A node is one true, specific,
    self-contained statement, worded objectively rather than about the
    user. Record a misconception as the correct statement, then record
    evidence that the user doesn't understand it.

    Args:
        statement: The true statement this node represents.
        depends_on: IDs of existing nodes that must be understood before
            this one can be.
    """

    fields, error = _require_text(statement=statement)
    if error:
        return error
    prerequisites = sorted(set(depends_on or []))

    def work(conn):
        missing = _missing_nodes(conn, prerequisites)
        if missing:
            return None, f"ERROR: no node with id {', '.join(map(str, missing))}."
        created = _timestamp()
        node = conn.execute(
            "INSERT INTO nodes (statement, created_at) VALUES (?, ?)",
            (fields["statement"], created),
        ).lastrowid
        conn.execute(
            "INSERT INTO nodes_fts (rowid, statement) VALUES (?, ?)",
            (node, fields["statement"]),
        )
        conn.executemany(
            "INSERT INTO dependencies (node_id, depends_on_id, created_at) VALUES (?, ?, ?)",
            [(node, prerequisite, created) for prerequisite in prerequisites],
        )
        return node, None

    node, error = _run(work)
    return error or f"Added node {node}."


@_learning_tool(study=True)
def add_dependency(node_id: int, depends_on_id: int, state=None) -> str:
    """Record that understanding one node requires understanding another.

    Args:
        node_id: The node that depends on the other.
        depends_on_id: The node that must be understood first.
    """

    if node_id == depends_on_id:
        return "ERROR: a node cannot depend on itself."

    def work(conn):
        missing = _missing_nodes(conn, (node_id, depends_on_id))
        if missing:
            return f"ERROR: no node with id {', '.join(map(str, missing))}."
        if _depends_transitively(conn, depends_on_id, node_id):
            return (
                f"ERROR: node {depends_on_id} already depends on node {node_id}, "
                "so this link would create a cycle."
            )
        conn.execute(
            "INSERT OR IGNORE INTO dependencies (node_id, depends_on_id, created_at)"
            " VALUES (?, ?, ?)",
            (node_id, depends_on_id, _timestamp()),
        )
        return f"Node {node_id} now depends on node {depends_on_id}."

    return _run(work)


@_learning_tool(study=True, writes={CYCLE}, init={CYCLE: dict})
def record_understanding(node_id: int, understands: bool, evidence: str, state=None) -> str:
    """Record evidence of whether the user understands a node.

    Call this after every answer, for each node the answer bears on, before
    calling next_study_step. If the current study step named a node, record
    evidence on that node; if it didn't, record the node your question asked
    about first. The harness records the time and a pointer to the user's
    latest message.

    Args:
        node_id: The node the evidence is about.
        understands: True if the user understands the node, False if not.
        evidence: What the user said or did that revealed knowledge or lack
            thereof on the node.
    """

    fields, error = _require_text(evidence=evidence)
    if error:
        return error
    if not isinstance(understands, bool):
        return "ERROR: understands must be true or false."

    # The first matching row after a question step is that question's answer.
    cycle = getattr(state, CYCLE, None) if state is not None else None
    phase = None
    if (
        isinstance(cycle, dict)
        and cycle.get("phase") in QUESTION_PHASES
        and cycle.get("answer_evidence_id") is None
        and cycle.get("node_id") in (None, node_id)
    ):
        phase = cycle["phase"]
    row = (
        node_id, int(understands), fields["evidence"], _timestamp(), *_pointer(state), phase,
    )

    def work(conn):
        if _missing_nodes(conn, (node_id,)):
            return None, f"ERROR: no node with id {node_id}."
        evidence_id = conn.execute(
            "INSERT INTO evidence (node_id, understands, evidence, observed_at, source_kind,"
            " project, session_id, user_entry_index, question_phase)"
            " VALUES (?, ?, ?, ?, 'study', ?, ?, ?, ?)",
            row,
        ).lastrowid
        return evidence_id, None

    evidence_id, error = _run(work)
    if error:
        return error
    if phase is not None:
        cycle["answer_evidence_id"] = evidence_id
        return f"Recorded as the answer to the current {phase} step."
    return "Recorded."


@_learning_tool(study=True)
def view_knowledge(node_id: int, state=None) -> str:
    """Show a node, its dependency links, and all evidence recorded against it.

    Args:
        node_id: The node to show.
    """

    now = _now()

    def work(conn):
        states = node_states(conn, now)
        node = states.get(node_id)
        if node is None:
            return f"ERROR: no node with id {node_id}."
        lines = [_describe(node), _describe_review(node, now)]
        lines.append("Depends on:")
        lines += [f"  {_describe(states[p])}" for p in node.depends_on] or ["  nothing"]
        lines.append("Depended on by:")
        lines += [f"  {_describe(states[d])}" for d in node.dependents] or ["  nothing"]
        lines.append("Evidence, newest first:")
        rows = conn.execute(
            "SELECT * FROM evidence WHERE node_id = ? ORDER BY observed_at DESC, id DESC",
            (node_id,),
        ).fetchall()
        for row in rows:
            verdict = "understands" if row["understands"] else "doesn't understand"
            source = row["source_kind"]
            if row["observation_id"] is not None:
                source += f" {row['observation_id']}"
            if row["question_phase"]:
                source += f", {row['question_phase']} answer"
            lines.append(
                f"  {row['observed_at'][:16]} {verdict} ({source}; {_format_pointer(row)}): "
                f"{row['evidence']}"
            )
        if not rows:
            lines.append("  none")
        return "\n".join(lines)

    return _run(work)


@_learning_tool(study=True)
def find_study_targets(limit: int = 8, state=None) -> str:
    """Rank the holes, edges, and nodes due for review in the knowledge graph.

    next_study_step already chooses targets; use this to explain the state
    of the graph to the user.

    A hole is a node that things the user understands depend on, but whose
    own evidence is thin, old, or negative. An edge is a node the user
    failed or hasn't been tested on, whose foundations look solid.

    Args:
        limit: Maximum number of nodes to return in each list.
    """

    limit = max(1, min(int(limit), 50))
    now = _now()
    cfg = load_config()

    def work(conn):
        states = list(node_states(conn, now, cfg).values())
        pending = len(conn.execute(_PENDING_OBSERVATIONS).fetchall())
        return states, pending

    states, pending = _run(work)
    holes = sorted((s for s in states if s.hole_score > 0), key=lambda s: -s.hole_score)
    hole_ids = {s.id for s in holes}
    edges = sorted(
        (s for s in states if s.edge_score > 0 and s.id not in hole_ids),
        key=lambda s: -s.edge_score,
    )
    due = sorted(
        (s for s in states if s.last_evidence and s.urgency >= cfg["review"]["due_ratio"]),
        key=lambda s: -s.review_priority,
    )

    lines = [f"Holes ({len(holes)}):"]
    lines += [
        f"  [{s.hole_score:.2f}] {_describe(s)}; {len(s.dependents)} dependent(s)"
        for s in holes[:limit]
    ] or ["  none"]
    lines.append(f"Edges ({len(edges)}):")
    lines += [f"  [{s.edge_score:.2f}] {_describe(s)}" for s in edges[:limit]] or ["  none"]
    lines.append(f"Due for review ({len(due)}):")
    lines += [
        f"  [{s.review_priority:.2f}] {_describe(s)}; {_describe_review(s, now)}"
        for s in due[:limit]
    ] or ["  none"]
    lines.append(f"Pending observations: {pending}")
    return "\n".join(lines)


@_learning_tool(study=True)
def list_observations(limit: int = 30, state=None) -> str:
    """List passive observations that working agents made about the user and
    that have not been attached to a node or dismissed, oldest first.

    Args:
        limit: Maximum number of observations to return.
    """

    limit = max(1, min(int(limit), 200))
    rows = _run(lambda conn: conn.execute(_PENDING_OBSERVATIONS + " LIMIT ?", (limit,)).fetchall())
    if not rows:
        return "No pending observations."
    return "\n".join(_describe_observation(row) for row in rows)


def _describe_observation(row: sqlite3.Row) -> str:
    verdict = "understands" if row["understands"] else "doesn't understand"
    return (
        f"observation {row['id']} ({row['created_at'][:16]}; {_format_pointer(row)}): "
        f"{verdict}: {row['topic']}\n  evidence: {row['evidence']}"
    )


@_learning_tool(study=True)
def attach_observation(observation_id: int, node_id: int, state=None) -> str:
    """Attach a passive observation to the node it concerns, recording it as
    evidence on that node. An observation may be attached to several nodes.

    Args:
        observation_id: The observation to attach.
        node_id: The node the observation is about.
    """

    def work(conn):
        obs = conn.execute(
            "SELECT * FROM observations WHERE id = ?", (observation_id,)
        ).fetchone()
        if obs is None:
            return f"ERROR: no observation with id {observation_id}."
        if _missing_nodes(conn, (node_id,)):
            return f"ERROR: no node with id {node_id}."
        try:
            conn.execute(
                "INSERT INTO evidence (node_id, understands, evidence, observed_at,"
                " source_kind, observation_id, project, session_id, user_entry_index)"
                " VALUES (?, ?, ?, ?, 'observation', ?, ?, ?, ?)",
                (
                    node_id, obs["understands"], obs["evidence"], obs["created_at"],
                    observation_id, obs["project"], obs["session_id"], obs["user_entry_index"],
                ),
            )
        except sqlite3.IntegrityError:
            return f"Observation {observation_id} is already attached to node {node_id}."
        return f"Attached observation {observation_id} to node {node_id}."

    return _run(work)


@_learning_tool(study=True)
def dismiss_observation(observation_id: int, state=None) -> str:
    """Dismiss a passive observation the user doesn't want to study.

    Args:
        observation_id: The observation to dismiss.
    """

    def work(conn):
        if conn.execute(
            "SELECT 1 FROM observations WHERE id = ?", (observation_id,)
        ).fetchone() is None:
            return f"ERROR: no observation with id {observation_id}."
        conn.execute(
            "INSERT OR IGNORE INTO observation_dismissals (observation_id, dismissed_at)"
            " VALUES (?, ?)",
            (observation_id, _timestamp()),
        )
        return f"Dismissed observation {observation_id}."

    return _run(work)


# ---------------------------------------------------------------------------
# The study cycle
#
# Each cycle reviews nodes that are due, probes until the user misses or the
# probe limit is reached, then teaches the missed node (or the next step past
# the last success) with a check question, reteaching after a missed check.
# The cycle position lives on session state; everything else is read from the
# graph.
# ---------------------------------------------------------------------------

def _cycle(state) -> dict:
    cycle = getattr(state, CYCLE, None)
    if not isinstance(cycle, dict):
        cycle = {}
        setattr(state, CYCLE, cycle)
    return cycle


def _new_cycle(cycle: dict) -> None:
    cycle.update(
        phase="", node_id=None, issued_at=None, answer_evidence_id=None,
        reviews_done=0, probes_done=0, teach_attempts=0, asked=[], last_probe=None,
    )


def _issue(cycle: dict, phase: str, node_id: int | None, now: datetime) -> None:
    cycle.update(
        phase=phase, node_id=node_id, issued_at=_timestamp(now), answer_evidence_id=None
    )


def _accuracy(conn, cfg, now) -> tuple[int, int, str | None]:
    acc = cfg["accuracy"]
    since = _timestamp(now - timedelta(hours=acc["window_hours"]))
    rows = conn.execute(
        "SELECT understands FROM evidence WHERE question_phase IS NOT NULL"
        " AND observed_at >= ? ORDER BY observed_at DESC, id DESC LIMIT ?",
        (since, acc["window_answers"]),
    ).fetchall()
    total = len(rows)
    correct = sum(row["understands"] for row in rows)
    mode = None
    if total and total >= acc["min_answers"]:
        if correct / total < acc["low"]:
            mode = "low"
        elif correct / total > acc["high"]:
            mode = "high"
    return correct, total, mode


def _accuracy_line(accuracy, cfg) -> str:
    correct, total, mode = accuracy
    if not total:
        return "No recent answers to judge difficulty from yet."
    line = f"Recent accuracy: {correct} of {total} ({correct / total:.0%})."
    if mode == "low":
        line += (
            f" That is below the {cfg['accuracy']['low']:.0%} target, so ask something"
            " easier than recent questions."
        )
    elif mode == "high":
        line += (
            f" That is above {cfg['accuracy']['high']:.0%}, so ask something harder"
            " than recent questions."
        )
    return line


def _probe_limit(cfg, accuracy) -> int:
    if accuracy[2] == "low":
        return 1
    return max(int(cfg["cycle"]["probe_questions"]), 1)


def _transition(cycle, phase, answer, cfg, accuracy, notes) -> tuple[str, dict | None]:
    """Record the outcome of the open step and return what comes next."""

    if answer is not None:
        node = answer["node_id"]
        understands = bool(answer["understands"])
    if phase == "review":
        cycle["reviews_done"] += 1
        if answer is not None:
            cycle["asked"].append(node)
            if not understands:
                return "teach", {"node_id": node, "reason": "review_miss"}
        return "review", None
    if phase == "probe":
        cycle["probes_done"] += 1
        limit = _probe_limit(cfg, accuracy)
        if answer is None:
            return ("probe" if cycle["probes_done"] < limit else "new"), None
        cycle["asked"].append(node)
        cycle["last_probe"] = {"node_id": node, "understands": understands}
        if not understands:
            return "teach", {"node_id": node, "reason": "probe_miss"}
        if cycle["probes_done"] >= limit:
            return "teach", {"node_id": None, "reason": "beyond", "beyond_node_id": node}
        return "probe", None
    if phase == "teach":
        cycle["teach_attempts"] += 1
        if answer is not None and not understands:
            if cycle["teach_attempts"] < max(int(cfg["cycle"]["max_teach_attempts"]), 1):
                return "teach", {"node_id": node, "reason": "reteach"}
            notes.append(
                f"Node {node} still hasn't landed. Give the correct answer, tell the user"
                " it will come back in review, and move on."
            )
        return "new", None
    if phase in ("orient", "focus"):
        return "probe", None
    return "new", None


def _pick_review(states, cycle, cfg) -> NodeState | None:
    asked = set(cycle["asked"])
    due = [
        s for s in states.values()
        if s.last_evidence is not None
        and s.id not in asked
        and s.urgency >= cfg["review"]["due_ratio"]
    ]
    return max(due, key=lambda s: (s.review_priority, -s.id), default=None)


def _neighbourhood(states, start: int, hops: int) -> set[int]:
    seen, frontier = {start}, {start}
    for _ in range(hops):
        frontier = {
            n for f in frontier for n in states[f].depends_on + states[f].dependents
        } - seen
        seen |= frontier
    return seen


def _pick_probe(conn, cycle, states, cfg, accuracy):
    """Return (node, observation, beyond) for the next probe.

    At most one is set. ``beyond`` is a node the user just got right that has
    nothing built on it yet, so the agent must add the next step itself.
    """

    asked = set(cycle["asked"])
    last = cycle.get("last_probe")
    if last and last["understands"] and last["node_id"] in states:
        base = states[last["node_id"]]
        harder = [states[d] for d in base.dependents if d not in asked]
        if harder:
            return max(harder, key=lambda s: (s.edge_score + s.hole_score, -s.weight)), None, None
        return None, None, base

    focus = cycle.get("focus_node_id")
    if focus in states:
        pool = _neighbourhood(states, focus, int(cfg["cycle"]["focus_hops"]))
    else:
        observation = conn.execute(_PENDING_OBSERVATIONS + " LIMIT 1").fetchone()
        if observation is not None:
            return None, observation, None
        pool = set(states)

    candidates = [states[n] for n in pool if n not in asked]
    if not candidates:
        return None, None, None
    if accuracy[2] == "low":
        key = lambda s: (s.known_below - s.doubt_below, -s.weight)  # noqa: E731
    else:
        key = lambda s: (s.edge_score + s.hole_score, -s.weight)  # noqa: E731
    return max(candidates, key=key), None, None


_FINISH_QUESTION = "End your turn after asking."


def _review_step(cycle, node, cfg, now, accuracy) -> list[str]:
    _issue(cycle, "review", node.id, now)
    return [
        f"Step: review (question {cycle['reviews_done'] + 1} of up to "
        f"{cfg['cycle']['review_questions']} this cycle)",
        _describe(node),
        _describe_review(node, now),
        _accuracy_line(accuracy, cfg),
        "Ask one question that makes the user recall or apply this node from memory,"
        " without giving the answer away. Prefer applying it to a new situation over"
        " restating it. " + _FINISH_QUESTION,
    ]


def _probe_step(conn, cycle, states, cfg, now, accuracy) -> list[str]:
    node, observation, beyond = _pick_probe(conn, cycle, states, cfg, accuracy)
    header = (
        f"Step: probe (question {cycle['probes_done'] + 1} of up to "
        f"{_probe_limit(cfg, accuracy)} before teaching)"
    )
    record_first = "Record the user's answer on that node first."
    if node is not None:
        _issue(cycle, "probe", node.id, now)
        lines = [
            header,
            _describe(node),
            "Ask one question about this node that finds where the user's understanding"
            " runs out.",
        ]
    elif observation is not None:
        _issue(cycle, "probe", None, now)
        lines = [
            header,
            _describe_observation(observation),
            "Find or add the node this observation concerns, attach the observation to it"
            " with attach_observation, and ask one question that tests it. " + record_first,
        ]
    elif beyond is not None:
        _issue(cycle, "probe", None, now)
        lines = [
            header,
            f"The user got node {beyond.id} right, and nothing in the graph builds on it"
            f" yet: {beyond.statement}",
            f"Ask about the next step beyond it. Add a node that depends on node"
            f" {beyond.id} if needed. " + record_first,
        ]
    elif not states:
        _issue(cycle, "orient", None, now)
        return [
            "Step: orient",
            "The knowledge graph is empty and there are no observations to test.",
            "Ask the user what they would like to study. When they answer, add a node for"
            " it with add_knowledge, call set_study_focus with its id, and then call"
            " next_study_step.",
        ]
    else:
        _issue(cycle, "probe", None, now)
        lines = [
            header,
            "Every nearby node has been asked about this cycle.",
            "Ask about something new near the focus, or near what the user has studied"
            " recently. Search the graph and add a node if needed. " + record_first,
        ]
    lines.append(_accuracy_line(accuracy, cfg))
    lines.append("Decide the correct answer before asking. " + _FINISH_QUESTION)
    return lines


_TEACH_FORMATS = {
    "worked example": (
        "This material is new to the user, so teach with a worked example: walk through"
        " one concrete instance step by step, then state the general point."
    ),
    "guided problem": (
        "The user's foundations here are solid, so explain briefly and make the check"
        " question a small problem that needs this step, with a hint."
    ),
}


def _teach_format(node: NodeState | None, beyond: NodeState | None) -> str:
    if node is None:
        return "guided problem" if beyond is not None and beyond.net > 0 else "worked example"
    if node.net > 0 or (node.known_below > 0 and node.doubt_below == 0):
        return "guided problem"
    return "worked example"


def _teach_step(cycle, states, now, teach) -> list[str]:
    node = states.get(teach["node_id"]) if teach["node_id"] is not None else None
    beyond = states.get(teach.get("beyond_node_id"))
    if teach["reason"] != "reteach":
        cycle["teach_attempts"] = 0
    _issue(cycle, "teach", node.id if node else None, now)

    reason = teach["reason"]
    lines = ["Step: teach"]
    if reason == "probe_miss":
        lines.append(
            f"The user just missed node {node.id}. Say briefly what was wrong in their"
            " answer, then teach it."
        )
    elif reason == "review_miss":
        lines.append(
            f"The user missed node {node.id} in review. Correct it briefly, then teach it"
            " again."
        )
    elif reason == "reteach":
        lines.append(
            f"The user missed the check on node {node.id} too. Teach it again from a"
            " different angle, more simply."
        )
    elif beyond is not None:
        lines.append(
            f"The user answered every probe correctly, ending at node {beyond.id}:"
            f" {beyond.statement}"
        )
        lines.append(
            f"Teach the next step beyond it. First add a node for that step that depends"
            f" on node {beyond.id}."
        )
    if node is not None:
        lines.append(_describe(node))
        known = [states[p] for p in node.depends_on if states[p].net > 0]
    else:
        known = [beyond] if beyond is not None else []
    if known:
        lines.append(
            "Build from what the user already holds: "
            + "; ".join(f"node {k.id}: {k.statement}" for k in known)
        )
    lines.append(_TEACH_FORMATS[_teach_format(node, beyond)])
    lines.append(
        "Teach one reasoning step and keep it short. End the message with one check"
        " question on the node you taught, then end your turn."
        + ("" if node is not None else " Record the answer on that node first.")
    )
    return lines


def _advance(conn, cycle, cfg, now, *, skip: bool) -> str:
    if "asked" not in cycle:
        _new_cycle(cycle)
    phase = cycle.get("phase") or ""
    notes = [f"Config warning: {warning}" for warning in cfg["_warnings"]]

    answer = None
    if phase in QUESTION_PHASES:
        if cycle.get("answer_evidence_id") is not None:
            answer = conn.execute(
                "SELECT * FROM evidence WHERE id = ?", (cycle["answer_evidence_id"],)
            ).fetchone()
        if answer is None and not skip:
            target = cycle.get("node_id")
            what = f"node {target}" if target is not None else "the node your question asked about"
            return (
                f"ERROR: the user's answer to the current {phase} step hasn't been recorded."
                f" Call record_understanding on {what} first, or call next_study_step with"
                " skip=true if the user wants to move on."
            )

    issued = _parse_time(cycle.get("issued_at") or "")
    if issued is not None:
        minutes = (now - issued).total_seconds() / 60.0
        if minutes >= cfg["cycle"]["recap_after_minutes"]:
            notes.append(
                f"It has been {minutes:.0f} minutes since the last step. Open with a"
                " one-line reminder of where you were."
            )

    states = node_states(conn, now, cfg)
    accuracy = _accuracy(conn, cfg, now)
    want, teach = _transition(cycle, phase, answer, cfg, accuracy, notes)
    if answer is not None and want != "teach":
        verdict = "correct" if answer["understands"] else "incorrect"
        notes.append(
            f"The user's last answer was recorded as {verdict}. Open with a sentence or"
            " two of feedback, and say what is true if they missed it."
        )

    if want == "new":
        _new_cycle(cycle)
        want = "review"
    if want == "review":
        node = None
        if cycle["reviews_done"] < cfg["cycle"]["review_questions"]:
            node = _pick_review(states, cycle, cfg)
        lines = (
            _review_step(cycle, node, cfg, now, accuracy)
            if node is not None
            else _probe_step(conn, cycle, states, cfg, now, accuracy)
        )
    elif want == "probe":
        lines = _probe_step(conn, cycle, states, cfg, now, accuracy)
    else:
        lines = _teach_step(cycle, states, now, teach)
    return "\n".join(notes + lines)


@_learning_tool(study=True, writes={CYCLE}, init={CYCLE: dict})
def next_study_step(skip: bool = False, state=None) -> str:
    """Get the next step of the study session: what to do, and which node.

    Call this after recording evidence for the user's last answer, and do
    exactly what the step says. The plugin decides when to review, probe,
    and teach, so you never need to plan the session yourself.

    Args:
        skip: True if the user wants to move on without answering the
            current question.
    """

    if state is None:
        return "ERROR: next_study_step needs session state."
    cfg = load_config()
    cycle = _cycle(state)
    return _run(lambda conn: _advance(conn, cycle, cfg, _now(), skip=bool(skip)))


@_learning_tool(study=True, writes={CYCLE}, init={CYCLE: dict})
def set_study_focus(node_id: int | None = None, state=None) -> str:
    """Focus the study session on a node and the nodes near it in the graph.

    Call this when the user names something they want to study, after
    finding or adding a node for it. Omit node_id to clear the focus.

    Args:
        node_id: The node to focus on.
    """

    if state is None:
        return "ERROR: set_study_focus needs session state."
    cycle = _cycle(state)
    if node_id is None:
        cycle["focus_node_id"] = None
        return "Focus cleared."
    if _run(lambda conn: _missing_nodes(conn, (node_id,))):
        return f"ERROR: no node with id {node_id}."
    _new_cycle(cycle)
    cycle["focus_node_id"] = node_id
    cycle["phase"] = "focus"
    return f"Focus set to node {node_id}. Call next_study_step for the first question."


STUDY_TOOLS = (
    next_study_step,
    set_study_focus,
    record_understanding,
    search_knowledge,
    add_knowledge,
    add_dependency,
    view_knowledge,
    find_study_targets,
    list_observations,
    attach_observation,
    dismiss_observation,
)


# ---------------------------------------------------------------------------
# Study-session instructions
# ---------------------------------------------------------------------------

STUDY_PROMPT = """

# Study session
This is a study session. The user studies here in the gaps while their other
agents work, so they may leave and come back at any moment. The plugin runs
the session: it decides when to review, probe, and teach, and which node to
work on. You carry out one step at a time and build the knowledge graph as you
go. The user should never have to organize their own studying.

## Each turn
1. If the user just answered a question, call record_understanding for every
   node the answer bears on. If the step named a node, record evidence on that
   node; if it didn't, record the node your question asked about first.
2. Call next_study_step and do exactly what it says.
3. Ask at most one question per message, then end your turn.

If the user asks you something or wants to discuss a point, answer them and
leave the current step open; call next_study_step only after they answer its
question. If they want to skip a question, call next_study_step with
skip=true. If they name something they want to study, find or add a node for
it and call set_study_focus.

## Grading
Decide the correct answer before you ask, and grade strictly against it. The
user answers by thinking out loud; read all of it. Their reasoning often shows
understanding or misunderstanding of nodes besides the one you asked about,
and their confidence shows in how they phrase it. A right answer reached
through confused reasoning is evidence against the reasoning's premises.
"I don't know" is a gap, not a wrong answer.

## The graph
Each node is one true, specific, self-contained statement, worded objectively
rather than about the user. A misconception enters the graph as the correct
statement, and the user's wrong belief is recorded as evidence that they don't
understand it. A node depends on another when understanding it requires
understanding the other. Before adding a node, call search_knowledge and reuse
an existing node that states the same idea in other words. Add dependency links
whenever you see them.

## Grounding
Observations and evidence point to a project, session, and entry index. When
one points at the user's own work, read that part of the transcript with the
trace tools if they are available, and ask about the user's actual code or
problem rather than a textbook example.

Keep every message short, because the user reads it between other tasks.
""".rstrip()


class StudyPromptInstructions(RenderTransformRegistrar):
    """Append study-session instructions to the system prompt while studying."""

    reads = {"entries"}
    writes = {"entries"}
    optional_reads = {STUDY_FLAG}

    @staticmethod
    def _make_transform(entry):
        def transform(messages, state, *, _entry=entry):
            if not getattr(state, STUDY_FLAG, False):
                return messages
            if not messages or _entry.role != "system":
                return messages
            first = dict(messages[0])
            if first.get("role") != "system":
                return messages
            content = str(first.get("content") or "")
            if "# Study session" in content:
                return messages
            first["content"] = content + STUDY_PROMPT
            return [first, *messages[1:]]

        return transform

    def __call__(self, state):
        if not getattr(state, "entries", None):
            return state
        entry = state.entries[0]
        entry.register_render_transform(
            MODEL_RENDER_CHANNEL,
            self._make_transform(entry),
            name="learning study-session system prompt",
        )
        return state


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _study_on(ctx):
    setattr(ctx.state, STUDY_FLAG, True)
    return CommandResult.notice(
        "Study session on. Say anything to start, or name a topic to study."
    )


def _study_off(ctx):
    setattr(ctx.state, STUDY_FLAG, False)
    return CommandResult.notice("Study session off.")


def register_features(builder, *, session, config):
    builder.add(Feature(
        name="learning_observations",
        components=[characterize_user_understanding],
    ))
    builder.add(Feature(
        name="learning_study",
        components=[*STUDY_TOOLS, StudyPromptInstructions()],
        order=[StudyPromptInstructions, ExcludeForgotten, MessageRenderer],
    ))
    builder.add(Feature(
        name="learning_commands",
        components=[
            Command(
                "/study", "Turn this session into a study session", _study_on,
                init={STUDY_FLAG: bool}, section="Learning",
            ),
            Command(
                "/study off", "End the study session", _study_off,
                init={STUDY_FLAG: bool}, section="Learning",
            ),
        ],
        order=[Command],
    ))
