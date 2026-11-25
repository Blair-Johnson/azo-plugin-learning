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

from collections import deque
import copy
import logging
import math
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from agent_utils.core.events import (
    Event,
    IoCompleted,
    IoFailed,
    IoInterrupted,
    RenderRequested,
    SessionRestored,
)
from agent_utils.core.io import IoRequest, OpId
from agent_utils.core.pipeline import EmitFn, Feature
from agent_utils.core.state import DurableSpec, MODEL_RENDER_CHANNEL, Entry, State
from agent_utils.core.tool import Tool
from agent_utils.features.core import (
    ExcludeForgotten,
    IoFence,
    MessageRenderer,
    RenderTransformRegistrar,
)
from agent_utils.features.interrupts import InterruptQueued
from agent_utils.runtime.driver import Driver
from agent_zoo.core.feature_state import session_state
from agent_zoo.durability import PluginStorage, StoreUpdate
from agent_zoo.events import BeforeLLM, EmitNotice
from agent_zoo.features.commands import Arg, Command, CommandContext, CommandResult
from agent_zoo.features.settings import Setting, SettingError
from agent_zoo.storage.projects import resolve_state_root

log = logging.getLogger("agent_zoo_user_plugin.learning")

_PLUGIN_NAME = "azo-plugin-learning"
_CONFIG_REL_PATH = "config/learning.yaml"
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
    return PluginStorage(
        _PLUGIN_NAME,
        config_format="yaml",
        config_filename=_CONFIG_REL_PATH,
    ).config.path


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


def _overlay_config_dict(raw: Any, *, source_label: str = "learning.yaml") -> dict[str, Any]:
    cfg: dict[str, Any] = {name: dict(values) for name, values in DEFAULT_CONFIG.items()}
    warnings: list[str] = []
    cfg["_warnings"] = warnings
    if raw is None:
        return cfg
    if not isinstance(raw, dict):
        warnings.append(f"{source_label} is not a mapping")
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


def load_config() -> dict[str, Any]:
    """Return defaults overlaid with the user's config file (driver/build-time use only)."""
    path = config_path()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return _overlay_config_dict({})
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        cfg = _overlay_config_dict({})
        cfg["_warnings"].append(f"could not read {path}: {exc}")
        return cfg
    return _overlay_config_dict(raw, source_label=str(path))


class _LearningConfigBox:
    """Pure in-memory container for learning settings loaded once at feature build time."""

    def __init__(
        self,
        raw_document: dict[str, Any],
        values: dict[str, Any],
        *,
        load_error: str | None = None,
    ) -> None:
        self.raw_document: dict[str, Any] = copy.deepcopy(raw_document)
        self.values: dict[str, Any] = copy.deepcopy(values)
        self.load_error: str | None = load_error

    def get_value(self, option: ConfigOption) -> int | str:
        val = self.values[option.section][option.key]
        return int(val) if isinstance(option.default, int) else format(val, "g")

    def set_value(self, option: ConfigOption, raw_value: Any, ctx: Any) -> None:
        integer = isinstance(option.default, int)
        candidate = raw_value
        if not integer:
            try:
                candidate = float(str(raw_value).strip())
            except ValueError:
                raise SettingError(f"{option.label} must be a number") from None
        try:
            coerced = coerce_option(option, candidate)
        except ValueError as exc:
            raise SettingError(f"{option.label} {exc}") from None
        if self.load_error is not None:
            raise SettingError(self.load_error)

        prev_raw = copy.deepcopy(self.raw_document)
        prev_values = copy.deepcopy(self.values)

        section_map = self.raw_document.get(option.section)
        if section_map is None:
            section_map = {}
            self.raw_document[option.section] = section_map
        elif not isinstance(section_map, dict):
            raise SettingError(
                f"{option.section} is not a valid YAML mapping; fix it before editing here"
            )
        section_map[option.key] = coerced
        self.values[option.section][option.key] = coerced
        updated_doc = copy.deepcopy(self.raw_document)

        def _rollback() -> None:
            self.raw_document = prev_raw
            self.values = prev_values

        if ctx is not None and callable(getattr(ctx, "submit_persist_io", None)):
            ctx.submit_persist_io(
                StoreUpdate(
                    plugin=_PLUGIN_NAME,
                    scope="config",
                    config_filename=_CONFIG_REL_PATH,
                    value=updated_doc,
                ),
                rollback=_rollback,
            )


def _load_config_box() -> _LearningConfigBox:
    """Build-time YAML loader populating an in-memory ``_LearningConfigBox``."""
    path = config_path()
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _LearningConfigBox({}, _overlay_config_dict({}))
    except (OSError, UnicodeError):
        return _LearningConfigBox(
            {},
            _overlay_config_dict({}),
            load_error=f"Cannot read {path}",
        )
    try:
        parsed = yaml.safe_load(raw_text)
    except yaml.YAMLError:
        return _LearningConfigBox(
            {},
            _overlay_config_dict({}),
            load_error=f"{path} is not a valid YAML mapping; fix it before editing here",
        )
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        return _LearningConfigBox(
            {},
            _overlay_config_dict({}),
            load_error=f"{path} is not a valid YAML mapping; fix it before editing here",
        )
    return _LearningConfigBox(parsed, _overlay_config_dict(parsed, source_label=str(path)))


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
# Per-session durable & runtime state + transcript pointers
# ---------------------------------------------------------------------------


@dataclass
class LearningState:
    """Durable per-session study mode and cycle position."""

    study_active: bool = False
    cycle: dict[str, Any] = field(default_factory=dict)


LEARNING_DURABLE_SPEC = DurableSpec(
    name="learning",
    cls=LearningState,
    version=1,
    on_fork="reset",
)


@dataclass
class LearningRuntime:
    """Strictly ephemeral per-session runtime tracking for in-flight /study ops."""

    pending_study_ops: dict[str, str] = field(default_factory=dict)


def _learning_state(state: Any) -> LearningState:
    if state is None:
        return LearningState()
    durable_fn = getattr(state, "durable", None)
    if callable(durable_fn):
        return durable_fn(LEARNING_DURABLE_SPEC)
    existing = getattr(state, "_learning_state_fallback", None)
    if not isinstance(existing, LearningState):
        existing = LearningState()
        try:
            setattr(state, "_learning_state_fallback", existing)
        except Exception:
            pass
    return existing


def _learning_runtime(state: Any) -> LearningRuntime:
    if state is None:
        return LearningRuntime()
    rt_fn = getattr(state, "runtime", None)
    if callable(rt_fn):
        return rt_fn("learning", LearningRuntime)
    existing = getattr(state, "_learning_runtime_fallback", None)
    if not isinstance(existing, LearningRuntime):
        existing = LearningRuntime()
        try:
            setattr(state, "_learning_runtime_fallback", existing)
        except Exception:
            pass
    return existing


def _current_session_id(state: Any) -> str:
    if state is None:
        return ""
    smeta = session_state(state)
    sid = str(smeta.session_id or "").strip()
    if sid:
        return sid
    context = getattr(state, "_agent_zoo_context", None)
    if isinstance(context, dict):
        return str(context.get("session_id") or "").strip()
    return ""


def _current_project(state: Any) -> str:
    if state is None:
        return ""
    smeta = session_state(state)
    proj = str(smeta.project_name or "").strip()
    if proj:
        return proj
    context = getattr(state, "_agent_zoo_context", None)
    if isinstance(context, dict):
        return str(context.get("project_name") or "").strip()
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
# Typed IoRequests, Results, and Commit-Class LearningDriver
# ---------------------------------------------------------------------------


@dataclass
class LearningRequest(IoRequest):
    """Base class for all LearningDriver SQLite/config IO requests."""


@dataclass
class CharacterizeUnderstandingOp(LearningRequest):
    topic: str
    evidence: str
    understands: bool
    project: str = ""
    session_id: str = ""
    user_entry_index: int | None = None
    timestamp: str = ""


@dataclass
class SearchKnowledgeOp(LearningRequest):
    query: str
    limit: int = 10


@dataclass
class AddKnowledgeOp(LearningRequest):
    statement: str
    depends_on: tuple[int, ...] = ()
    timestamp: str = ""


@dataclass
class AddDependencyOp(LearningRequest):
    node_id: int
    depends_on_id: int
    timestamp: str = ""


@dataclass
class RecordUnderstandingOp(LearningRequest):
    node_id: int
    understands: bool
    evidence: str
    project: str = ""
    session_id: str = ""
    user_entry_index: int | None = None
    answers: bool = False
    cycle: dict[str, Any] | None = None
    timestamp: str = ""


@dataclass(frozen=True)
class RecordUnderstandingResult:
    text: str
    cycle: dict[str, Any] | None = None


@dataclass
class ViewKnowledgeOp(LearningRequest):
    node_id: int


@dataclass
class AttachObservationOp(LearningRequest):
    observation_id: int
    node_id: int


@dataclass
class DismissObservationOp(LearningRequest):
    observation_id: int
    timestamp: str = ""


@dataclass
class StudyPickTopicOp(LearningRequest):
    message: str = ""
    cycle: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StudyPickTopicResult:
    step: str
    cycle: dict[str, Any]


class LearningDriver(Driver):
    """Commit-class serialized SQLite driver for all learning graph & observation I/O."""

    durability = "commit"
    request_types = (LearningRequest,)

    def __init__(self, *, db_file: Path | None = None) -> None:
        super().__init__()
        self._db_file = db_file
        self._cond = threading.Condition(threading.RLock())
        self._queue: deque[tuple[OpId, LearningRequest]] = deque()
        self._cancelled: set[OpId] = set()
        self._in_progress = False
        self._closing = False
        self._closed = False
        self._worker: threading.Thread | None = None

    def start(self, emit: EmitFn) -> None:
        super().start(emit)
        with self._cond:
            self._closing = False
            self._closed = False
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._worker_loop,
                    name="agent-zoo-learning-db",
                    daemon=True,
                )
                self._worker.start()

    def submit(self, op_id: OpId, req: IoRequest) -> None:
        if not isinstance(req, LearningRequest):
            raise TypeError(
                f"LearningDriver expected LearningRequest, got {type(req).__name__}"
            )
        with self._cond:
            if self._closing or self._closed or op_id in self._cancelled:
                return
            self._queue.append((op_id, req))
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._worker_loop,
                    name="agent-zoo-learning-db",
                    daemon=True,
                )
                self._worker.start()
            self._cond.notify_all()

    def cancel(self, op_id: OpId) -> None:
        with self._cond:
            self._cancelled.add(op_id)

    def drain(self, *, timeout_s: float | None = None) -> None:
        deadline = (
            None
            if timeout_s is None
            else (time.monotonic() + max(0.0, float(timeout_s)))
        )
        with self._cond:
            while (self._queue or self._in_progress) and not self._closed:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    self._cond.wait(timeout=min(remaining, 0.25))
                else:
                    self._cond.wait(timeout=0.25)

    def quiesce(
        self,
        *,
        policy: str = "terminate",
        deadline: float | None = None,
    ) -> list[Event]:
        del policy
        remaining = (
            None
            if deadline is None
            else max(0.0, float(deadline) - time.monotonic())
        )
        self.drain(timeout_s=remaining)
        return []

    def close(self) -> None:
        with self._cond:
            self._closing = True
            worker = self._worker
            self._cond.notify_all()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=5.0)
        with self._cond:
            self._closed = True
            self._worker = None
            self._cancelled.clear()
            self._cond.notify_all()

    def _is_fenced(self, op_id: OpId) -> bool:
        with self._cond:
            return self._closed or op_id in self._cancelled

    def _run_db(self, work):
        conn = connect(self._db_file)
        try:
            with conn:
                return work(conn)
        finally:
            conn.close()

    def execute_request(self, req: LearningRequest) -> Any:
        """Execute one typed ``LearningRequest`` against SQLite (called on worker thread)."""
        if isinstance(req, CharacterizeUnderstandingOp):
            ts = req.timestamp or _timestamp()
            row = (
                ts,
                req.topic,
                req.evidence,
                int(req.understands),
                req.project,
                req.session_id,
                req.user_entry_index,
            )
            self._run_db(
                lambda conn: conn.execute(
                    "INSERT INTO observations (created_at, topic, evidence, understands,"
                    " project, session_id, user_entry_index) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    row,
                )
            )
            return "Recorded."

        if isinstance(req, SearchKnowledgeOp):
            terms = re.findall(r"\w+", str(req.query or ""))
            if not terms:
                return "ERROR: query must contain words to search for."
            match = " OR ".join('"' + term + '"' for term in terms)
            limit = max(1, min(int(req.limit), 50))
            cfg = load_config()

            def _search(conn):
                ids = [
                    row["rowid"]
                    for row in conn.execute(
                        "SELECT rowid FROM nodes_fts WHERE nodes_fts MATCH ?"
                        " ORDER BY bm25(nodes_fts) LIMIT ?",
                        (match, limit),
                    )
                ]
                states = node_states(conn, cfg=cfg)
                return [states[node] for node in ids if node in states]

            found = self._run_db(_search)
            if not found:
                return "No matching nodes."
            return "\n".join(_describe(ns) for ns in found)

        if isinstance(req, AddKnowledgeOp):
            prerequisites = sorted(set(req.depends_on))
            created = req.timestamp or _timestamp()

            def _add(conn):
                missing = _missing_nodes(conn, prerequisites)
                if missing:
                    return None, f"ERROR: no node with id {', '.join(map(str, missing))}."
                node = conn.execute(
                    "INSERT INTO nodes (statement, created_at) VALUES (?, ?)",
                    (req.statement, created),
                ).lastrowid
                conn.execute(
                    "INSERT INTO nodes_fts (rowid, statement) VALUES (?, ?)",
                    (node, req.statement),
                )
                conn.executemany(
                    "INSERT INTO dependencies (node_id, depends_on_id, created_at) VALUES (?, ?, ?)",
                    [(node, prerequisite, created) for prerequisite in prerequisites],
                )
                return node, None

            node, err = self._run_db(_add)
            return err or f"Added node {node}."

        if isinstance(req, AddDependencyOp):
            created = req.timestamp or _timestamp()

            def _dep(conn):
                missing = _missing_nodes(conn, (req.node_id, req.depends_on_id))
                if missing:
                    return f"ERROR: no node with id {', '.join(map(str, missing))}."
                if _depends_transitively(conn, req.depends_on_id, req.node_id):
                    return (
                        f"ERROR: node {req.depends_on_id} already depends on node {req.node_id}, "
                        "so this link would create a cycle."
                    )
                conn.execute(
                    "INSERT OR IGNORE INTO dependencies (node_id, depends_on_id, created_at)"
                    " VALUES (?, ?, ?)",
                    (req.node_id, req.depends_on_id, created),
                )
                return f"Node {req.node_id} now depends on node {req.depends_on_id}."

            return self._run_db(_dep)

        if isinstance(req, RecordUnderstandingOp):
            cycle = copy.deepcopy(req.cycle) if isinstance(req.cycle, dict) else None
            phase = cycle.get("phase") if (req.answers and cycle is not None) else None
            ts = req.timestamp or _timestamp()
            row = (
                req.node_id,
                int(req.understands),
                req.evidence,
                ts,
                req.project,
                req.session_id,
                req.user_entry_index,
                phase if phase in QUESTION_PHASES else None,
            )
            cfg = load_config()

            def _record(conn):
                if _missing_nodes(conn, (req.node_id,)):
                    return RecordUnderstandingResult(
                        text=f"ERROR: no node with id {req.node_id}.",
                        cycle=cycle,
                    )
                evidence_id = conn.execute(
                    "INSERT INTO evidence (node_id, understands, evidence, observed_at, source_kind,"
                    " project, session_id, user_entry_index, question_phase)"
                    " VALUES (?, ?, ?, ?, 'study', ?, ?, ?, ?)",
                    row,
                ).lastrowid
                if not req.answers or cycle is None:
                    return RecordUnderstandingResult(text="Recorded.", cycle=cycle)
                cycle["answered_entry"] = req.user_entry_index
                answer = conn.execute(
                    "SELECT * FROM evidence WHERE id = ?", (evidence_id,)
                ).fetchone()
                text = "Recorded. Next:\n" + _advance(conn, cycle, cfg, _now(), answer)
                return RecordUnderstandingResult(text=text, cycle=cycle)

            return self._run_db(_record)

        if isinstance(req, ViewKnowledgeOp):
            now = _now()
            cfg = load_config()

            def _view(conn):
                states = node_states(conn, now, cfg)
                node = states.get(req.node_id)
                if node is None:
                    return f"ERROR: no node with id {req.node_id}."
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
                    (req.node_id,),
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

            return self._run_db(_view)

        if isinstance(req, AttachObservationOp):
            def _attach(conn):
                obs = conn.execute(
                    "SELECT * FROM observations WHERE id = ?", (req.observation_id,)
                ).fetchone()
                if obs is None:
                    return f"ERROR: no observation with id {req.observation_id}."
                if _missing_nodes(conn, (req.node_id,)):
                    return f"ERROR: no node with id {req.node_id}."
                try:
                    conn.execute(
                        "INSERT INTO evidence (node_id, understands, evidence, observed_at,"
                        " source_kind, observation_id, project, session_id, user_entry_index)"
                        " VALUES (?, ?, ?, ?, 'observation', ?, ?, ?, ?)",
                        (
                            req.node_id,
                            obs["understands"],
                            obs["evidence"],
                            obs["created_at"],
                            req.observation_id,
                            obs["project"],
                            obs["session_id"],
                            obs["user_entry_index"],
                        ),
                    )
                except sqlite3.IntegrityError:
                    return f"Observation {req.observation_id} is already attached to node {req.node_id}."
                return f"Attached observation {req.observation_id} to node {req.node_id}."

            return self._run_db(_attach)

        if isinstance(req, DismissObservationOp):
            ts = req.timestamp or _timestamp()

            def _dismiss(conn):
                if (
                    conn.execute(
                        "SELECT 1 FROM observations WHERE id = ?", (req.observation_id,)
                    ).fetchone()
                    is None
                ):
                    return f"ERROR: no observation with id {req.observation_id}."
                conn.execute(
                    "INSERT OR IGNORE INTO observation_dismissals (observation_id, dismissed_at)"
                    " VALUES (?, ?)",
                    (req.observation_id, ts),
                )
                return f"Dismissed observation {req.observation_id}."

            return self._run_db(_dismiss)

        if isinstance(req, StudyPickTopicOp):
            cycle = copy.deepcopy(req.cycle) if isinstance(req.cycle, dict) else {}
            if "asked" not in cycle:
                _new_cycle(cycle)
            cfg = load_config()
            if req.message:
                step = self._run_db(lambda conn: _directed_step(conn, cycle, cfg, _now()))
            else:
                step = self._run_db(lambda conn: _advance(conn, cycle, cfg, _now(), None))
            return StudyPickTopicResult(step=step, cycle=cycle)

        raise TypeError(f"Unsupported LearningRequest: {type(req).__name__}")

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue:
                    if self._closing or self._closed:
                        return
                    self._cond.wait(timeout=0.5)
                op_id, req = self._queue.popleft()
                cancelled = op_id in self._cancelled
                if not cancelled:
                    self._in_progress = True

            if cancelled:
                with self._cond:
                    self._cancelled.discard(op_id)
                    self._cond.notify_all()
                continue

            try:
                value = self.execute_request(req)
                if not self._is_fenced(op_id):
                    self._safe_emit(IoCompleted(op_id=op_id, value=value))
            except Exception as exc:
                log.warning("LearningDriver: %s failed: %s", type(req).__name__, exc)
                if not self._is_fenced(op_id):
                    self._safe_emit(IoFailed(op_id=op_id, error=exc))
            finally:
                with self._cond:
                    self._in_progress = False
                    self._cancelled.discard(op_id)
                    self._cond.notify_all()


# ---------------------------------------------------------------------------
# Tools (100% pure Sans-I/O generator tools yielding LearningRequest)
# ---------------------------------------------------------------------------

_STATE_READS = {
    "entries",
    "_agent_zoo_context",
}


class LearningTool(Tool):
    """A tool shown only inside, or only outside, study sessions."""

    study: bool = True
    optional_reads: set[str] = set(_STATE_READS)

    def available(self, state: State) -> bool:
        return bool(_learning_state(state).study_active) == self.study


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
    ):
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

        project, sid, entry_idx = _pointer(state)
        outcome = yield CharacterizeUnderstandingOp(
            topic=fields["topic"],
            evidence=fields["evidence"],
            understands=understands,
            project=project,
            session_id=sid,
            user_entry_index=entry_idx,
            timestamp=_timestamp(),
        )
        return str(outcome)


characterize_user_understanding = CharacterizeUserUnderstanding()


class SearchKnowledge(LearningTool):
    name = "search_knowledge"
    study = True

    def call(self, state: State | None, query: str, limit: int = 10):
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
        outcome = yield SearchKnowledgeOp(
            query=str(query),
            limit=max(1, min(int(limit), 50)),
        )
        return str(outcome)


search_knowledge = SearchKnowledge()


class AddKnowledge(LearningTool):
    name = "add_knowledge"
    study = True

    def call(
        self,
        state: State | None,
        statement: str,
        depends_on: list[int] | None = None,
    ):
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
        prerequisites = tuple(sorted(set(depends_on or [])))
        outcome = yield AddKnowledgeOp(
            statement=fields["statement"],
            depends_on=prerequisites,
            timestamp=_timestamp(),
        )
        return str(outcome)


add_knowledge = AddKnowledge()


class AddDependency(LearningTool):
    name = "add_dependency"
    study = True

    def call(self, state: State | None, node_id: int, depends_on_id: int):
        """Record that understanding one node requires understanding another.

        Args:
            node_id: The node that depends on the other.
            depends_on_id: The node that must be understood first.
        """

        if node_id == depends_on_id:
            return "ERROR: a node cannot depend on itself."
        outcome = yield AddDependencyOp(
            node_id=node_id,
            depends_on_id=depends_on_id,
            timestamp=_timestamp(),
        )
        return str(outcome)


add_dependency = AddDependency()


class RecordUnderstanding(LearningTool):
    name = "record_understanding"
    study = True

    def call(
        self,
        state: State | None,
        node_id: int,
        understands: bool,
        evidence: str,
    ):
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

        project, sid, entry_idx = _pointer(state)
        lst = _learning_state(state) if state is not None else None
        cycle = copy.deepcopy(lst.cycle) if lst is not None else None
        answers = (
            lst is not None
            and lst.study_active
            and (entry_idx is None or entry_idx != lst.cycle.get("answered_entry"))
        )
        outcome = yield RecordUnderstandingOp(
            node_id=node_id,
            understands=understands,
            evidence=fields["evidence"],
            project=project,
            session_id=sid,
            user_entry_index=entry_idx,
            answers=answers,
            cycle=cycle,
            timestamp=_timestamp(),
        )
        if isinstance(outcome, RecordUnderstandingResult):
            if lst is not None and outcome.cycle is not None:
                lst.cycle = outcome.cycle
            return outcome.text
        return str(outcome)


record_understanding = RecordUnderstanding()


class ViewKnowledge(LearningTool):
    name = "view_knowledge"
    study = True

    def call(self, state: State | None, node_id: int):
        """Show a node, its dependency links, and all evidence recorded against it.

        Args:
            node_id: The node to show.
        """

        outcome = yield ViewKnowledgeOp(node_id=node_id)
        return str(outcome)


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

    def call(self, state: State | None, observation_id: int, node_id: int):
        """Attach a passive observation to the node it concerns, recording it as
        evidence on that node. An observation may be attached to several nodes.

        Args:
            observation_id: The observation to attach.
            node_id: The node the observation is about.
        """

        outcome = yield AttachObservationOp(
            observation_id=observation_id,
            node_id=node_id,
        )
        return str(outcome)


attach_observation = AttachObservation()


class DismissObservation(LearningTool):
    name = "dismiss_observation"
    study = True

    def call(self, state: State | None, observation_id: int):
        """Dismiss a passive observation the user doesn't want to study.

        Args:
            observation_id: The observation to dismiss.
        """

        outcome = yield DismissObservationOp(
            observation_id=observation_id,
            timestamp=_timestamp(),
        )
        return str(outcome)


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
    return _learning_state(state).cycle


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

    @staticmethod
    def _make_transform(entry):
        def transform(messages, state, *, _entry=entry):
            if not _learning_state(state).study_active:
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
# Commands & Pure Sans-I/O Study Session Handler
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


class LearningStudyHandler:
    """Pure Sans-I/O component handling ``StudyPickTopicOp`` completions for ``/study``."""

    handles = (IoCompleted, IoFailed, IoInterrupted, SessionRestored)

    def handle(self, event: Event, state: State, emit: EmitFn) -> None:
        rt = _learning_runtime(state)
        if isinstance(event, SessionRestored):
            rt.pending_study_ops.clear()
            return
        if event.op_id not in rt.pending_study_ops:
            return
        message = rt.pending_study_ops.pop(event.op_id)
        lst = _learning_state(state)

        if isinstance(event, IoCompleted):
            val = event.value
            if isinstance(val, StudyPickTopicResult):
                step = val.step
                lst.cycle = dict(val.cycle)
            else:
                step = str(val or "")
            if message:
                _append_entry(state, STUDY_START_MESSAGE + step, system_generated=True)
                _append_entry(state, _user_message(message))
                lst.cycle["answered_entry"] = _latest_user_entry_index(state)
                emit(RenderRequested())
                emit(BeforeLLM())
            else:
                pending = list(getattr(state, "pending_interrupts", None) or [])
                pending.append(STUDY_START_MESSAGE + step)
                state.pending_interrupts = pending
                lst.cycle["answered_entry"] = _latest_user_entry_index(state)
                emit(InterruptQueued(source="learning"))
                emit(RenderRequested())
                emit(BeforeLLM())
            return

        # IoFailed or IoInterrupted
        lst.study_active = False
        reason = (
            str(event.error)
            if isinstance(event, IoFailed)
            else str(event.reason or "interrupted")
        )
        emit(EmitNotice(text=f"Could not start study session: {reason}", level="warning"))
        emit(RenderRequested())


def _study_on(ctx: CommandContext, message: str = "") -> CommandResult:
    message = str(message or "").strip()
    lst = _learning_state(ctx.state)
    if lst.study_active:
        if message:
            _append_entry(ctx.state, _user_message(message))
            return CommandResult.notice(
                "Study session is already on; sent your message.",
                events=(RenderRequested(), BeforeLLM()),
                force_follow=True,
            )
        return CommandResult.notice("Study session is already on.", refresh=False)

    lst.study_active = True
    _new_cycle(lst.cycle)
    lst.cycle["phase"] = ""

    req = StudyPickTopicOp(message=message, cycle=copy.deepcopy(lst.cycle))
    submit_fn = getattr(ctx, "submit_io", None)
    if callable(submit_fn):
        op_id = submit_fn(req, tag="learning_study")
    else:
        op_id = ctx.state.io.submit(req, tag="learning_study")
    _learning_runtime(ctx.state).pending_study_ops[op_id] = message
    return CommandResult.notice(
        "Study session on.",
        events=(RenderRequested(),),
        force_follow=True,
    )


def _study_off(ctx: CommandContext) -> CommandResult:
    _learning_state(ctx.state).study_active = False
    return CommandResult.notice("Study session off.", events=(RenderRequested(),))


# ---------------------------------------------------------------------------
# /config settings (pure Sans-I/O via in-memory _LearningConfigBox + StoreUpdate)
# ---------------------------------------------------------------------------

_SETTING_GROUPS = {
    "cycle": "Learning / Study cycle",
    "accuracy": "Learning / Difficulty",
    "review": "Learning / Review timing",
    "scoring": "Learning / Scoring",
}


def _make_setting(option: ConfigOption, box: _LearningConfigBox) -> Setting:
    # The settings API has no float kind, so decimal options are edited as text.
    integer = isinstance(option.default, int)

    return Setting(
        key=f"learning.{option.section}.{option.key}",
        label=option.label,
        kind="int" if integer else "text",
        get=lambda ctx, _box=box, _opt=option: _box.get_value(_opt),
        set=lambda ctx, value, _box=box, _opt=option: _box.set_value(_opt, value, ctx),
        group=_SETTING_GROUPS[option.section],
        help=f"{option.help} Default: {option.default:g}.",
        min=option.minimum if integer else None,
        max=option.maximum if integer else None,
    )


def register_features(builder, *, session, config):
    del session, config
    box = _load_config_box()
    builder.add(Feature(
        name="learning_observations",
        components=[characterize_user_understanding],
    ))
    builder.add(Feature(
        name="learning_study",
        components=[*STUDY_TOOLS, StudyPromptInstructions(), LearningStudyHandler()],
        order=[IoFence, LearningStudyHandler, StudyPromptInstructions, ExcludeForgotten, MessageRenderer],
        drivers=[LearningDriver],
        durable_state=LEARNING_DURABLE_SPEC,
    ))
    builder.add(Feature(
        name="learning_commands",
        components=[
            Command(
                "/study", "Turn this session into a study session", _study_on,
                args=(Arg("message", kind="text", default=""),),
                usage="/study [what you'd like to study]",
                section="Learning",
            ),
            Command(
                "/study off", "End the study session", _study_off,
                section="Learning",
            ),
        ],
        order=[Command],
    ))
    builder.add(Feature(
        name="learning_settings",
        components=[_make_setting(option, box) for option in CONFIG_OPTIONS],
        order=[Setting],
    ))
