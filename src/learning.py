"""Learning add-on: a persistent model of what the user understands.

Working agents record passive observations with
``characterize_user_understanding``. Those observations land in a global queue.

``/study`` turns the current session into a study session. Study sessions see
a different tool set: they build a knowledge graph of true statements linked
by dependencies, and record evidence of understanding against its nodes.
The plugin keeps the session's rhythm, a repeating cycle of review, probing,
and teaching: recording the user's answer returns the next step together with
the graph context behind it. The agent chooses what to ask and teach by
reading the graph and the user's answers.

The graph is the only learning data structure. Nothing stores a schedule or a
mastery score: each node's review timing, hole and edge scores are derived from
its evidence and its neighbours' evidence whenever they are read. Review is a
ranking, so it can never build up a backlog of overdue items.

Everything lives in one SQLite database under the Agent Zoo state home, shared
by every project and session. Thresholds come from the plugin's YAML config.
"""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from agent_utils.core.events import Event, RenderRequested
from agent_utils.core.pipeline import EmitFn, Feature
from agent_utils.core.state import MODEL_RENDER_CHANNEL, Entry, State
from agent_utils.core.tool import Tool
from agent_utils.features.core import (
    ExcludeForgotten,
    MessageRenderer,
    RenderTransformRegistrar,
)
from agent_utils.features.interrupts import InterruptQueued
from agent_zoo.events import BeforeLLM
from agent_zoo.features.commands import Arg, Command, CommandContext, CommandResult
from agent_zoo.features.settings import Setting, SettingError
from agent_zoo.storage.projects import resolve_state_root

_PLUGIN_NAME = "azo-plugin-learning"
STUDY_FLAG = "learning_study_active"
CYCLE = "learning_study_cycle"
QUESTION_PHASES = ("review", "probe", "teach", "orient")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigOption:
    section: str
    key: str
    default: int | float
    label: str
    help: str
    minimum: int | float = 0
    maximum: int | float | None = None


# Every threshold the plugin reads. Integer defaults make integer options.
CONFIG_OPTIONS = (
    ConfigOption(
        "cycle", "review_questions", 2, "Review questions per cycle",
        "Questions on nodes due for review at the start of each cycle, when any are due.",
    ),
    ConfigOption(
        "cycle", "probe_questions", 3, "Probe questions per cycle",
        "Probe questions before teaching, unless the user misses one first.", minimum=1,
    ),
    ConfigOption(
        "cycle", "max_teach_attempts", 2, "Teaching attempts",
        "Times a missed node is taught before it is left for review.", minimum=1,
    ),
    ConfigOption(
        "cycle", "focus_hops", 2, "Focus distance",
        "How many dependency links from the last answered node to look for probe candidates.",
    ),
    ConfigOption(
        "cycle", "candidates", 6, "Candidates per step",
        "Candidate nodes and observations shown to the agent in each step.", minimum=1,
    ),
    ConfigOption(
        "accuracy", "window_answers", 10, "Answers considered",
        "Most recent graded answers used to judge whether questions are too hard or easy.",
        minimum=1,
    ),
    ConfigOption(
        "accuracy", "window_hours", 12, "Accuracy window (hours)",
        "Only answers from this many recent hours count toward accuracy.",
    ),
    ConfigOption(
        "accuracy", "min_answers", 4, "Minimum answers",
        "Answers needed in the window before difficulty adjusts.",
    ),
    ConfigOption(
        "accuracy", "low", 0.75, "Low accuracy threshold",
        "Below this fraction correct, cycles probe once and the agent asks easier questions.",
        maximum=1,
    ),
    ConfigOption(
        "accuracy", "high", 0.90, "High accuracy threshold",
        "Above this fraction correct, the agent asks harder questions.", maximum=1,
    ),
    ConfigOption(
        "review", "base_interval_days", 1.0, "Base interval (days)",
        "Review interval after a miss or a first success.",
    ),
    ConfigOption(
        "review", "growth_factor", 3.0, "Interval growth",
        "Each further spaced success multiplies the review interval by this.",
    ),
    ConfigOption(
        "review", "min_success_gap_hours", 8.0, "Minimum spacing (hours)",
        "Successes closer together than this count as one.",
    ),
    ConfigOption(
        "review", "due_ratio", 1.0, "Due ratio",
        "A node is due once the time since its last evidence reaches this multiple of its"
        " interval. Due nodes are ranked, never queued.",
    ),
    ConfigOption(
        "review", "dependent_weight", 0.25, "Dependent bonus",
        "Extra review priority for each node that depends on this one.",
    ),
    ConfigOption(
        "scoring", "half_life_days", 30.0, "Evidence half-life (days)",
        "Evidence weight halves over this many days when scoring holes and edges.",
    ),
    ConfigOption(
        "scoring", "observation_weight", 0.4, "Observation weight",
        "Weight of a passive observation from a working session, relative to a study answer.",
    ),
    ConfigOption(
        "scoring", "neighbour_weight", 0.5, "Neighbour weight",
        "How much neighbouring nodes' evidence counts toward hole and edge scores.",
    ),
)

_OPTIONS = {(option.section, option.key): option for option in CONFIG_OPTIONS}

DEFAULT_CONFIG: dict[str, dict[str, float | int]] = {}
for _option in CONFIG_OPTIONS:
    DEFAULT_CONFIG.setdefault(_option.section, {})[_option.key] = _option.default


def config_path() -> Path:
    return resolve_state_root() / "plugin-configs" / _PLUGIN_NAME / "config" / "learning.yaml"


def coerce_option(option: ConfigOption, value: Any) -> int | float:
    """Return ``value`` as the option's type, or raise ValueError."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a number")
    if not math.isfinite(value):
        raise ValueError("must be finite")
    if isinstance(option.default, int):
        if value != int(value):
            raise ValueError("must be a whole number")
        value = int(value)
    else:
        value = float(value)
    if value < option.minimum:
        raise ValueError(f"must be at least {option.minimum:g}")
    if option.maximum is not None and value > option.maximum:
        raise ValueError(f"must be at most {option.maximum:g}")
    return value


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
    for section in DEFAULT_CONFIG:
        values = raw.get(section) or {}
        if not isinstance(values, dict):
            warnings.append(f"{section} is not a mapping")
            continue
        for key, value in values.items():
            option = _OPTIONS.get((section, key))
            if option is None:
                warnings.append(f"unknown setting {section}.{key}")
                continue
            try:
                cfg[section][key] = coerce_option(option, value)
            except ValueError as exc:
                warnings.append(f"{section}.{key} {exc}")
    return cfg


def _set_yaml_value(text: str, section: str, key: str, scalar: str) -> str:
    """Set ``section.key`` in a simple block-style YAML file, keeping comments."""

    lines = text.splitlines()
    header = re.compile(rf"^{re.escape(section)}\s*:\s*(#.*)?$")
    start = next((i for i, line in enumerate(lines) if header.match(line)), None)
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines += [f"{section}:", f"  {key}: {scalar}"]
        return "\n".join(lines) + "\n"

    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t#"):
        end += 1
    entry = re.compile(rf"^(\s+){re.escape(key)}\s*:(.*)$")
    for i in range(start + 1, end):
        match = entry.match(lines[i])
        if match:
            comment = match.group(2).find(" #")
            suffix = match.group(2)[comment:] if comment >= 0 else ""
            lines[i] = f"{match.group(1)}{key}: {scalar}{suffix}"
            return "\n".join(lines) + "\n"
    insert = end
    while insert > start + 1 and not lines[insert - 1].strip():
        insert -= 1
    lines.insert(insert, f"  {key}: {scalar}")
    return "\n".join(lines) + "\n"


def _replace_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_config_value(option: ConfigOption, value: int | float) -> None:
    """Save one option to the config file, leaving everything else unchanged.

    The edit is checked by rereading the whole config. If any other value or
    warning changed, the original file is restored and SettingError is raised.
    """

    path = config_path()
    try:
        original = path.read_text() if path.exists() else None
    except OSError:
        raise SettingError(f"Cannot read {path}") from None
    try:
        parsed = yaml.safe_load(original or "") or {}
    except yaml.YAMLError:
        parsed = None
    if not isinstance(parsed, dict):
        raise SettingError(f"{path} is not a valid YAML mapping; fix it before editing here")

    before = load_config()
    updated = _set_yaml_value(original or "", option.section, option.key, repr(value))
    try:
        _replace_file(path, updated)
        after = load_config()
        expected = {name: dict(values) for name, values in before.items() if name != "_warnings"}
        expected[option.section][option.key] = value
        if {name: values for name, values in after.items() if name != "_warnings"} != expected:
            raise ValueError("the saved file did not read back as expected")
        own = f"{option.section}.{option.key} "
        if [w for w in after["_warnings"] if not w.startswith(own)] != [
            w for w in before["_warnings"] if not w.startswith(own)
        ]:
            raise ValueError("the saved file produced new warnings")
    except (OSError, ValueError):
        try:
            if original is None:
                path.unlink(missing_ok=True)
            else:
                _replace_file(path, original)
        except OSError:
            pass
        raise SettingError(f"Could not save {option.label} to {path}") from None


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
    sid = str(getattr(state, "_session_id", "") or "").strip()
    if sid:
        return sid
    context = getattr(state, "_agent_zoo_context", None)
    if isinstance(context, dict):
        return str(context.get("session_id") or "").strip()
    return ""


def _current_project(state: Any) -> str:
    for attr in ("_project_name", "project_name"):
        value = str(getattr(state, attr, "") or "").strip()
        if value:
            return value
    context = getattr(state, "_agent_zoo_context", None)
    if isinstance(context, dict):
        value = str(context.get("project_name") or "").strip()
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

_STATE_READS = {
    "entries",
    "_session_id",
    "_agent_zoo_context",
    "run_db",
    "_project_name",
    STUDY_FLAG,
    CYCLE,
}


class LearningTool(Tool):
    """A tool shown only inside, or only outside, study sessions."""

    study: bool = True
    optional_reads: set[str] = set(_STATE_READS)

    def available(self, state: State) -> bool:
        return bool(getattr(state, STUDY_FLAG, False)) == self.study


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


class CharacterizeUserUnderstanding(LearningTool):
    name = "characterize_user_understanding"
    study = False

    def call(
        self,
        state: State | None,
        topic: str,
        evidence: str,
        understands: bool,
    ) -> str:
        """Characterize whether the user understands something, with brief evidence.

        Call this whenever the user reveals what they do or don't understand,
        even if it has nothing to do with your current task. That includes asking
        what something means, asking for an explanation, seeming unsure, saying
        something incorrect, explaining a concept correctly without help, or
        catching a mistake you made. It returns immediately and does not affect
        your work; you do not need to mention it to the user.

        Only record knowledge about the wider world that is worth the user
        internalizing: how a technology, algorithm, model architecture, protocol,
        mathematical idea, or scientific concept works. Do not record knowledge
        of the code, files, configuration, or project details on this machine,
        such as what a local function does or where a setting lives.

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


characterize_user_understanding = CharacterizeUserUnderstanding()


class SearchKnowledge(LearningTool):
    name = "search_knowledge"
    study = True

    def call(self, state: State | None, query: str, limit: int = 10) -> str:
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
        return "\n".join(_describe(node_state) for node_state in found)


search_knowledge = SearchKnowledge()


class AddKnowledge(LearningTool):
    name = "add_knowledge"
    study = True

    def call(
        self,
        state: State | None,
        statement: str,
        depends_on: list[int] | None = None,
    ) -> str:
        """Add a node to the user's knowledge graph.

        Search with search_knowledge first. A node is one true, specific,
        self-contained statement about the wider world, worded objectively
        rather than about the user, never about code or files on this machine. Record a misconception as the correct statement, then record
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


add_knowledge = AddKnowledge()


class AddDependency(LearningTool):
    name = "add_dependency"
    study = True

    def call(self, state: State | None, node_id: int, depends_on_id: int) -> str:
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


add_dependency = AddDependency()


class RecordUnderstanding(LearningTool):
    name = "record_understanding"
    study = True
    writes = {CYCLE}
    init = {CYCLE: dict}

    def call(
        self,
        state: State | None,
        node_id: int,
        understands: bool,
        evidence: str,
    ) -> str:
        """Record evidence of whether the user understands a node.

        Call this after every answer, for each node the answer bears on. Record
        the node your question asked about first: the first record for each
        user message is graded as their answer, and its result tells you what to
        do next. The harness records the time and a pointer to the user's latest
        message.

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

        pointer = _pointer(state)
        cycle = _cycle(state) if state is not None else None
        # The first record for each user message is the answer that advances the
        # cycle; later records on the same message are extra evidence.
        answers = (
            cycle is not None
            and getattr(state, STUDY_FLAG, False)
            and (pointer[2] is None or pointer[2] != cycle.get("answered_entry"))
        )
        phase = cycle.get("phase") if answers else None
        row = (
            node_id, int(understands), fields["evidence"], _timestamp(), *pointer,
            phase if phase in QUESTION_PHASES else None,
        )
        cfg = load_config()

        def work(conn):
            if _missing_nodes(conn, (node_id,)):
                return f"ERROR: no node with id {node_id}."
            evidence_id = conn.execute(
                "INSERT INTO evidence (node_id, understands, evidence, observed_at, source_kind,"
                " project, session_id, user_entry_index, question_phase)"
                " VALUES (?, ?, ?, ?, 'study', ?, ?, ?, ?)",
                row,
            ).lastrowid
            if not answers:
                return "Recorded."
            cycle["answered_entry"] = pointer[2]
            answer = conn.execute("SELECT * FROM evidence WHERE id = ?", (evidence_id,)).fetchone()
            return "Recorded. Next:\n" + _advance(conn, cycle, cfg, _now(), answer)

        return _run(work)


record_understanding = RecordUnderstanding()


class ViewKnowledge(LearningTool):
    name = "view_knowledge"
    study = True

    def call(self, state: State | None, node_id: int) -> str:
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
            lines = [f"{_flag(node)}{_describe(node)}", _describe_review(node, now)]
            lines.append("Depends on:")
            lines += [
                f"  {_flag(states[p])}{_describe(states[p])}" for p in node.depends_on
            ] or ["  nothing"]
            lines.append("Depended on by:")
            lines += [
                f"  {_flag(states[d])}{_describe(states[d])}" for d in node.dependents
            ] or ["  nothing"]
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


view_knowledge = ViewKnowledge()


def _describe_observation(row: sqlite3.Row) -> str:
    verdict = "understands" if row["understands"] else "doesn't understand"
    return (
        f"observation {row['id']} ({row['created_at'][:16]}; {_format_pointer(row)}): "
        f"{verdict}: {row['topic']}\n  evidence: {row['evidence']}"
    )


class AttachObservation(LearningTool):
    name = "attach_observation"
    study = True

    def call(self, state: State | None, observation_id: int, node_id: int) -> str:
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


attach_observation = AttachObservation()


class DismissObservation(LearningTool):
    name = "dismiss_observation"
    study = True

    def call(self, state: State | None, observation_id: int) -> str:
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


dismiss_observation = DismissObservation()


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
        phase="", node_id=None,
        reviews_done=0, probes_done=0, teach_attempts=0, asked=[], last_probe=None,
    )


def _issue(cycle: dict, phase: str, node_id: int | None) -> None:
    cycle.update(phase=phase, node_id=node_id)


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
    """Count the answer against the open step and return the next phase."""

    node = answer["node_id"] if answer is not None else None
    understands = bool(answer["understands"]) if answer is not None else True
    if node is not None:
        cycle["asked"].append(node)
        # The session's focus follows whatever the user last answered.
        cycle["anchor_node_id"] = node
    if phase == "review":
        cycle["reviews_done"] += 1
        if not understands:
            return "teach", {"node_id": node, "reason": "review_miss"}
        return "review", None
    if phase in ("probe", "orient"):
        cycle["probes_done"] += 1
        cycle["last_probe"] = {"node_id": node, "understands": understands}
        if not understands:
            return "teach", {"node_id": node, "reason": "probe_miss"}
        if cycle["probes_done"] >= _probe_limit(cfg, accuracy):
            return "teach", {"node_id": node, "reason": "beyond"}
        return "probe", None
    if phase == "teach":
        cycle["teach_attempts"] += 1
        if not understands:
            if cycle["teach_attempts"] < max(int(cfg["cycle"]["max_teach_attempts"]), 1):
                return "teach", {"node_id": node, "reason": "reteach"}
            notes.append(
                f"Node {node} still hasn't landed. Give the correct answer, tell the user"
                " it will come back in review, and move on."
            )
        return "new", None
    return "new", None


def _limit(cfg) -> int:
    return max(int(cfg["cycle"]["candidates"]), 1)


def _due(states, cycle, cfg) -> list[NodeState]:
    asked = set(cycle["asked"])
    due = [
        s for s in states.values()
        if s.last_evidence is not None
        and s.id not in asked
        and s.urgency >= cfg["review"]["due_ratio"]
    ]
    return sorted(due, key=lambda s: (-s.review_priority, s.id))[: _limit(cfg)]


def _neighbourhood(states, start: int, hops: int) -> set[int]:
    seen, frontier = {start}, {start}
    for _ in range(hops):
        frontier = {
            n for f in frontier for n in states[f].depends_on + states[f].dependents
        } - seen
        seen |= frontier
    return seen


def _flag(state: NodeState) -> str:
    if state.hole_score > 0 and state.hole_score >= state.edge_score:
        return f"possible hole [{state.hole_score:.2f}]; "
    if state.edge_score > 0:
        return f"possible edge [{state.edge_score:.2f}]; "
    return ""


def _candidate_lines(states, cycle, cfg) -> tuple[str, list[str]]:
    """Nodes worth considering for the next probe, with what the graph knows."""

    asked = set(cycle["asked"])
    anchor = cycle.get("anchor_node_id")
    if anchor in states:
        pool = _neighbourhood(states, anchor, int(cfg["cycle"]["focus_hops"])) - asked
        heading = f"Nodes near node {anchor}, where the user last answered:"
    else:
        pool = {s.id for s in states.values() if s.hole_score > 0 or s.edge_score > 0} - asked
        heading = "Possible holes and edges across the graph:"
    ranked = sorted(
        (states[n] for n in pool),
        key=lambda s: (-(s.hole_score + s.edge_score), s.weight, s.id),
    )[: _limit(cfg)]
    return heading, [f"  {_flag(s)}{_describe(s)}" for s in ranked]


_FINISH_QUESTION = "End your turn after asking."


def _review_step(cycle, due, cfg, now, accuracy) -> list[str]:
    _issue(cycle, "review", None)
    return [
        f"Step: review (question {cycle['reviews_done'] + 1} of up to "
        f"{cfg['cycle']['review_questions']} this cycle)",
        "These nodes are due for review, most urgent first:",
        *[f"  {_describe(s)}; {_describe_review(s, now)}" for s in due],
        "Choose one to review. Prefer a node connected to what the user has been"
        " studying, or one that other nodes rest on. Ask a question that makes the user"
        " recall or apply it from memory, without giving the answer away; applying it"
        " to a new situation beats restating it.",
        _accuracy_line(accuracy, cfg),
        _FINISH_QUESTION,
    ]


def _probe_step(conn, cycle, states, cfg, accuracy) -> list[str]:
    observations = conn.execute(
        _PENDING_OBSERVATIONS + " LIMIT ?", (_limit(cfg),)
    ).fetchall()
    if not states and not observations:
        _issue(cycle, "orient", None)
        return [
            "Step: orient",
            "The knowledge graph is empty and there are no observations to test.",
            "Ask the user what they would like to study. When they answer, find or add a"
            " node for it and ask one question that finds where their understanding of it"
            " runs out. " + _FINISH_QUESTION,
        ]

    _issue(cycle, "probe", None)
    lines = [
        f"Step: probe (question {cycle['probes_done'] + 1} of up to "
        f"{_probe_limit(cfg, accuracy)} before teaching)"
    ]
    last = cycle.get("last_probe")
    if last and last["understands"] and last["node_id"] in states:
        base = states[last["node_id"]]
        lines.append(f"The user just answered node {base.id} correctly: {base.statement}")
        built = [states[d] for d in base.dependents if d not in cycle["asked"]]
        if built:
            lines.append("Nodes built on it:")
            lines += [f"  {_flag(s)}{_describe(s)}" for s in built[: _limit(cfg)]]
        lines.append(
            "Ask something a step harder than what they just showed. If the graph doesn't"
            " have that step yet, add it."
        )
    else:
        heading, candidates = _candidate_lines(states, cycle, cfg)
        if candidates:
            lines.append(heading)
            lines += candidates
        if observations:
            lines.append("Untested observations from the user's work:")
            lines += [f"  {_describe_observation(row)}" for row in observations]
            lines.append(
                "To test an observation, find or add the node it concerns and attach it"
                " with attach_observation."
            )
        lines.append(
            "Choose what to probe by reading these, the user's recent answers, and the"
            " graph around them with view_knowledge and search_knowledge. Ask one question"
            " that finds where the user's understanding runs out."
        )
    lines.append(_accuracy_line(accuracy, cfg))
    lines.append("Decide the correct answer before asking. " + _FINISH_QUESTION)
    return lines


def _teach_step(cycle, states, teach) -> list[str]:
    reason = teach["reason"]
    node = states.get(teach["node_id"])
    if reason != "reteach":
        cycle["teach_attempts"] = 0
    _issue(cycle, "teach", None)

    lines = ["Step: teach"]
    if node is None:
        lines.append("Teach the step the user's last answer showed they are missing.")
    elif reason == "probe_miss":
        lines.append(
            f"The user missed node {node.id}: {node.statement}\nSay briefly what was wrong,"
            " then teach the step they are missing. Usually that is this node, but if"
            " their reasoning showed a gap in something it rests on, teach that first."
        )
    elif reason == "review_miss":
        lines.append(
            f"The user missed node {node.id} in review: {node.statement}\nCorrect it"
            " briefly, then teach it again, or the prerequisite their answer showed they"
            " lack."
        )
    elif reason == "reteach":
        lines.append(
            f"The user missed the check on node {node.id} too: {node.statement}\nTeach it"
            " again from a different angle and more simply, or teach a prerequisite their"
            " answer showed they lack."
        )
    else:
        lines.append(
            f"The user answered every probe correctly, ending at node {node.id}:"
            f" {node.statement}\nTeach the next step beyond what they showed. Add a node"
            " for it if the graph doesn't have one."
        )
    if node is not None and node.depends_on:
        lines.append("It rests on:")
        lines += [f"  {_describe(states[p])}" for p in node.depends_on]
    lines.append(
        "If the material is new to the user, teach with a worked example: walk through"
        " one concrete instance, then state the general point. If they already hold its"
        " foundations, explain briefly and make the check a small problem that needs it."
    )
    lines.append(
        "Teach one reasoning step and keep it short. End with one check question on what"
        " you taught, then end your turn."
    )
    return lines


def _advance(conn, cycle, cfg, now, answer) -> str:
    """Count ``answer`` (an evidence row, or None) and describe the next step."""

    if "asked" not in cycle:
        _new_cycle(cycle)
    phase = cycle.get("phase") or ""
    notes = [f"Config warning: {warning}" for warning in cfg["_warnings"]]

    states = node_states(conn, now, cfg)
    accuracy = _accuracy(conn, cfg, now)
    want, teach = _transition(cycle, phase, answer, cfg, accuracy, notes)
    if answer is not None and want != "teach":
        verdict = "correct" if answer["understands"] else "incorrect"
        notes.append(
            f"The user's answer was recorded as {verdict}. Open with a sentence or two of"
            " feedback, and say what is true if they missed it."
        )

    if want == "new":
        _new_cycle(cycle)
        want = "review"
    if want == "review":
        due = []
        if cycle["reviews_done"] < cfg["cycle"]["review_questions"]:
            due = _due(states, cycle, cfg)
        lines = (
            _review_step(cycle, due, cfg, now, accuracy)
            if due
            else _probe_step(conn, cycle, states, cfg, accuracy)
        )
    elif want == "probe":
        lines = _probe_step(conn, cycle, states, cfg, accuracy)
    else:
        lines = _teach_step(cycle, states, teach)
    return "\n".join(notes + lines)


def _directed_step(conn, cycle, cfg, now) -> str:
    """The first step when the user's /study message says what to study."""

    accuracy = _accuracy(conn, cfg, now)
    _issue(cycle, "probe", None)
    return "\n".join([
        *(f"Config warning: {warning}" for warning in cfg["_warnings"]),
        f"Step: probe (question 1 of up to {_probe_limit(cfg, accuracy)} before teaching)",
        "The user's message below says what they want to study. Find it in the graph"
        " with search_knowledge and read what is there and around it with"
        " view_knowledge, including the evidence of what they already understand. Add"
        " a node for it if the graph has none. Then ask one question that finds where"
        " their understanding of it runs out.",
        _accuracy_line(accuracy, cfg),
        "Decide the correct answer before asking. " + _FINISH_QUESTION,
    ])


STUDY_TOOLS = (
    record_understanding,
    view_knowledge,
    search_knowledge,
    add_knowledge,
    add_dependency,
    attach_observation,
    dismiss_observation,
)


# ---------------------------------------------------------------------------
# Study-session instructions
# ---------------------------------------------------------------------------

STUDY_PROMPT = """

# Study session
This is a study session. The user studies here in the gaps while their other
agents work, so they may leave and come back at any moment. They should never
have to organize their own studying: the plugin keeps the rhythm of review,
probing, and teaching, and you decide what to ask and teach by reading the
knowledge graph and reacting to what the user shows you.

## Each turn
After the user answers, call record_understanding for every node the answer
bears on, starting with the node your question asked about. The first record
for each answer returns the next step: whether to review, probe, or teach,
with the graph context behind it. Follow the step, but choose its content
yourself. Read the candidate nodes, follow their dependencies with
view_knowledge, search for related nodes, and anchor your choice in what the
graph holds and what the user's answers revealed. Ask at most one question per
message, then end your turn.

If the user asks something or wants to discuss a point, answer them; the step
waits for their next answer. If they name something they want to study, find
or add a node for it and ask about it. If they don't know an answer, that is
evidence of a gap.

## Grading
Decide the correct answer before you ask, and grade strictly against it. The
user answers by thinking out loud; read all of it. Their reasoning often shows
understanding or misunderstanding of nodes besides the one you asked about,
and their confidence shows in how they phrase it. A right answer reached
through confused reasoning is evidence against the reasoning's premises.

## The graph
Each node is one true, specific, self-contained statement, worded objectively
rather than about the user. Nodes hold knowledge about the wider world: how a
technology, algorithm, model, mathematical idea, or scientific concept works.
Never add nodes about the code, files, or project details on the user's
machine. A misconception enters the graph as the correct
statement, and the user's wrong belief is recorded as evidence that they don't
understand it. A node depends on another when understanding it requires
understanding the other. Before adding a node, call search_knowledge and reuse
an existing node that states the same idea in other words. Add dependency links
whenever you see them.

Evidence summaries show how many records a node has, when the last was, and a
net score that is positive when recent evidence says the user understands it.
A possible hole is a node that well-understood nodes rest on but whose own
evidence is thin, old, or negative. A possible edge is a node the user failed
or hasn't been tested on whose foundations look solid.

## Grounding
Observations and evidence point to a project, session, and entry index. When
one points at the user's own work, read that part of the transcript with the
trace tools if they are available, and use the user's actual code or problem
as the setting for a question rather than a textbook example. The question
still tests the general knowledge, not the local code. If an observation is
about local details rather than knowledge of the world, dismiss it.

## Presentation
The user's terminal renders Markdown, LaTeX, and Mermaid. Write inline math as
$...$ and display math as $$...$$ on lines of its own. Draw diagrams in
```mermaid code blocks. Use these where they make an explanation or question
clearer: equations for anything mathematical, and diagrams for structure,
flows, or how ideas depend on each other. Keep every message short, because
the user reads it between other tasks.
""".rstrip()


class StudyPromptInstructions(RenderTransformRegistrar):
    """Append study-session instructions to the system prompt while studying."""

    handles = (RenderRequested,)
    reads = {"entries"}
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

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        assert isinstance(event, RenderRequested)
        if event.channels is not None and MODEL_RENDER_CHANNEL not in event.channels:
            return
        if not getattr(state, "entries", None):
            return
        entry = state.entries[0]
        entry.append_render_transform(
            MODEL_RENDER_CHANNEL,
            self._make_transform(entry),
            name="learning study-session system prompt",
        )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

STUDY_START_MESSAGE = (
    "[Harness]: The user ran /study to start a study session. The study tools"
    " are now available. Follow the study-session instructions, starting with"
    " this step:\n\n"
)


def _user_message(text: str) -> str:
    # Matches how the runtime frames typed user input.
    return f"--- USER ---\n{text}\n--- END USER ---"


def _append_entry(state: Any, content: str, *, system_generated: bool = False) -> Entry:
    entries = getattr(state, "entries", None)
    if entries is None:
        entries = []
        state.entries = entries
    entry = Entry(
        messages=[{"role": "user", "content": content}],
        index=len(entries),
        step=int(getattr(state, "step", 0) or 0),
        system_generated=system_generated,
    )
    entries.append(entry)
    state.dirty = True
    state.done = False
    return entry


def _study_on(ctx: CommandContext, message: str = "") -> CommandResult:
    message = str(message or "").strip()
    if getattr(ctx.state, STUDY_FLAG, False):
        if message:
            _append_entry(ctx.state, _user_message(message))
            return CommandResult.notice(
                "Study session is already on; sent your message.",
                events=(RenderRequested(), BeforeLLM()),
                force_follow=True,
            )
        return CommandResult.notice("Study session is already on.", refresh=False)
    setattr(ctx.state, STUDY_FLAG, True)

    cycle = _cycle(ctx.state)
    _new_cycle(cycle)
    cycle["phase"] = ""
    cfg = load_config()
    if message:
        step = _run(lambda conn: _directed_step(conn, cycle, cfg, _now()))
        # Queued interrupts are held until after a pending user message is
        # answered, so append both entries directly to keep them in order.
        _append_entry(ctx.state, STUDY_START_MESSAGE + step, system_generated=True)
        _append_entry(ctx.state, _user_message(message))
        events: tuple[Event, ...] = (RenderRequested(), BeforeLLM())
    else:
        step = _run(lambda conn: _advance(conn, cycle, cfg, _now(), None))
        # Interrupt delivery waits for a valid slot after any unanswered user
        # message, then starts the agent's turn.
        pending = list(getattr(ctx.state, "pending_interrupts", None) or [])
        pending.append(STUDY_START_MESSAGE + step)
        ctx.state.pending_interrupts = pending
        events = (InterruptQueued(source="learning"), RenderRequested(), BeforeLLM())
    # Messages up to now, including the /study message, are not answers.
    cycle["answered_entry"] = _latest_user_entry_index(ctx.state)
    return CommandResult.notice("Study session on.", events=events, force_follow=True)


def _study_off(ctx: CommandContext) -> CommandResult:
    setattr(ctx.state, STUDY_FLAG, False)
    return CommandResult.notice("Study session off.", events=(RenderRequested(),))


# ---------------------------------------------------------------------------
# /config settings
# ---------------------------------------------------------------------------

_SETTING_GROUPS = {
    "cycle": "Learning / Study cycle",
    "accuracy": "Learning / Difficulty",
    "review": "Learning / Review timing",
    "scoring": "Learning / Scoring",
}


def _make_setting(option: ConfigOption) -> Setting:
    # The settings API has no float kind, so decimal options are edited as text.
    integer = isinstance(option.default, int)

    def get(ctx):
        value = load_config()[option.section][option.key]
        return value if integer else format(value, "g")

    def set_(ctx, value):
        if not integer:
            try:
                value = float(str(value).strip())
            except ValueError:
                raise SettingError(f"{option.label} must be a number") from None
        try:
            value = coerce_option(option, value)
        except ValueError as exc:
            raise SettingError(f"{option.label} {exc}") from None
        write_config_value(option, value)

    return Setting(
        key=f"learning.{option.section}.{option.key}",
        label=option.label,
        kind="int" if integer else "text",
        get=get,
        set=set_,
        group=_SETTING_GROUPS[option.section],
        help=f"{option.help} Default: {option.default:g}.",
        min=option.minimum if integer else None,
        max=option.maximum if integer else None,
    )


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
                args=(Arg("message", kind="text", default=""),),
                usage="/study [what you'd like to study]",
                init={STUDY_FLAG: bool}, section="Learning",
            ),
            Command(
                "/study off", "End the study session", _study_off,
                init={STUDY_FLAG: bool}, section="Learning",
            ),
        ],
        order=[Command],
    ))
    builder.add(Feature(
        name="learning_settings",
        components=[_make_setting(option) for option in CONFIG_OPTIONS],
        order=[Setting],
    ))
