from __future__ import annotations

import inspect
import json
import queue
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from agent_utils.core.events import InputReceived, IoFailed, IoInterrupted, RenderRequested, SessionRestored
from agent_utils.core.io import SubmitOp
from agent_utils.core.pipeline import PipelineBuilder
from agent_utils.core.session import Session
from agent_utils.core.state import Entry, State
from agent_utils.features.core import core_feature
from agent_utils.features.interrupts import interrupts_feature
from agent_utils.runtime.loop import Runtime
from agent_utils.runtime.simulation import SimulationRuntime
from agent_utils.storage.serialization import capture_session_document, restore_session_document
from agent_zoo.core.feature_state import session_state
from agent_zoo.durability import PluginStoreDriver, StoreUpdate
from agent_zoo.events import ReplyFrontendRequest, SettingsRequest, SlashCommand
from agent_zoo.features.commands import Command, CommandRegistry, commands_feature
from agent_zoo.features.settings import Setting, settings_feature
from agent_zoo.pipelines.default import build_default_builder
from agent_zoo.testing.sans_io_guard import assert_pipeline_sans_io, audit_module_sans_io

import src.learning as plugin


@pytest.fixture(autouse=True)
def state_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ZOO_HOME", str(tmp_path))
    return tmp_path


def _rows(table):
    conn = plugin.connect()
    try:
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
    finally:
        conn.close()


def _entry(index, role, content, *, system_generated=False):
    return Entry(
        messages=[{"role": role, "content": content}],
        index=index,
        step=0,
        tokens=1,
        system_generated=system_generated,
    )


def _session_state(*entries, study=False):
    state = State()
    state.entries = list(entries)
    smeta = session_state(state)
    smeta.session_id = "session-1"
    smeta.project_name = "demo"
    state.durable(plugin.LEARNING_DURABLE_SPEC).study_active = study
    return state


def _run_tool_gen(tool, state, **kwargs):
    """Drive a pure Sans-I/O learning generator tool against ``LearningDriver``."""
    ret = tool.call(state, **kwargs)
    if not inspect.isgenerator(ret):
        return ret
    driver = plugin.LearningDriver()
    try:
        try:
            req = next(ret)
            while True:
                outcome = driver.execute_request(req)
                req = ret.send(outcome)
        except StopIteration as stop:
            return stop.value
    finally:
        driver.close()


def _build_session(*, with_interrupts=False, with_commands=False, with_settings=False):
    session = Session(system_prompt="Base system")
    builder = PipelineBuilder().add(core_feature())
    if with_interrupts:
        builder.add(interrupts_feature())
    if with_commands:
        builder.add(commands_feature())
    if with_settings:
        builder.add(settings_feature())
    plugin.register_features(builder, session=session, config={})
    session.pipeline = builder.build()
    session.ensure_initialized()
    return session, builder


def _make_sim(session):
    """Attach a ``SimulationRuntime`` backed by ``LearningDriver``."""
    driver = plugin.LearningDriver()
    sim = SimulationRuntime(session)
    sim.register_handler(
        plugin.LearningRequest,
        lambda _op_id, req, _sim: driver.execute_request(req),
    )
    return sim, driver


def _add(statement, depends_on=None):
    result = _run_tool_gen(plugin.add_knowledge, None, statement=statement, depends_on=depends_on)
    assert result.startswith("Added node "), result
    return int(result.removeprefix("Added node ").rstrip("."))


def _evidence(node, understands, *, days_ago=0.0, hours_ago=0.0, kind="study", phase=None):
    moment = datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)
    conn = plugin.connect()
    try:
        with conn:
            conn.execute(
                "INSERT INTO evidence (node_id, understands, evidence, observed_at,"
                " source_kind, question_phase) VALUES (?, ?, 'e', ?, ?, ?)",
                (node, int(understands), moment.isoformat(timespec="seconds"), kind, phase),
            )
    finally:
        conn.close()


def _states():
    conn = plugin.connect()
    try:
        return plugin.node_states(conn)
    finally:
        conn.close()


def _write_config(tmp_path, text):
    del tmp_path
    path = plugin.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _start(state=None):
    """Run /study through ``LearningStudyHandler`` + ``LearningDriver`` and return (state, first step)."""
    if state is None:
        state = State(entries=[_entry(0, "system", "sys")])
    if not hasattr(state, "pending_interrupts"):
        state.pending_interrupts = []
    handler = plugin.LearningStudyHandler()
    driver = plugin.LearningDriver()
    try:
        plugin._study_on(SimpleNamespace(state=state))
        effects = state.io.take_outbox()
        assert len(effects) == 1
        eff = effects[0]
        assert isinstance(eff, SubmitOp) and isinstance(eff.request, plugin.StudyPickTopicOp)
        val = driver.execute_request(eff.request)
        state.io.complete(eff.op_id)
        from agent_utils.core.events import IoCompleted

        handler.handle(IoCompleted(op_id=eff.op_id, value=val), state, lambda _ev: None)
    finally:
        driver.close()
    message = state.pending_interrupts[-1]
    assert message.startswith(plugin.STUDY_START_MESSAGE)
    return state, message.removeprefix(plugin.STUDY_START_MESSAGE)


def _record(state, node, understands):
    return _run_tool_gen(
        plugin.record_understanding,
        state,
        node_id=node,
        understands=understands,
        evidence="answer",
    )


def _answer(state, node, understands):
    """The user sends a message, and the agent records it on ``node``."""
    state.entries.append(_entry(len(state.entries), "user", "answer"))
    return _record(state, node, understands)


# --- Sans-I/O guard, driver contract, durability, pipeline ------------------


def test_sans_io_audit_driver_durability_and_session_persistence():
    assert audit_module_sans_io(Path(__file__).resolve().parents[1] / "src/learning.py") == []
    assert plugin.LearningDriver.durability == "commit"
    assert plugin.LearningDriver.request_types == (plugin.LearningRequest,)

    session, builder = _build_session(with_interrupts=True, with_commands=True, with_settings=True)
    drivers = builder.build_drivers()
    assert any(isinstance(d, plugin.LearningDriver) for d in drivers)
    for d in drivers:
        d.close()

    assert_pipeline_sans_io(
        session.pipeline,
        session=session,
        events=[RenderRequested(), SessionRestored(), SlashCommand("/study")],
    )

    # Durable LearningState round-trips through capture/restore cleanly.
    lst = session.state.durable("learning", plugin.LearningState)
    lst.study_active = True
    lst.cycle = {"phase": "probe", "probes_done": 1, "asked": [42]}
    doc = json.loads(json.dumps(capture_session_document(session.state, session.pipeline)))
    restored, _ = _build_session(with_interrupts=True, with_commands=True)
    restore_session_document(doc, restored.pipeline, state=restored.state)
    restored_lst = restored.state.durable("learning", plugin.LearningState)
    assert restored_lst.study_active is True
    assert restored_lst.cycle["phase"] == "probe"
    assert restored_lst.cycle["asked"] == [42]
    assert plugin.LEARNING_DURABLE_SPEC.on_fork == "reset"


def test_pipeline_builds_with_all_tools():
    session, _ = _build_session()

    assert plugin.characterize_user_understanding in session.pipeline
    for study_tool in plugin.STUDY_TOOLS:
        assert study_tool in session.pipeline


def test_composes_onto_default_agent_zoo_builder():
    session = Session(system_prompt="Base system")
    builder = build_default_builder(
        session, config={}, include_plugins=False, include_llm_drivers=False
    )
    plugin.register_features(builder, session=session, config={})
    session.pipeline = builder.build()
    session.ensure_initialized()
    assert_pipeline_sans_io(session.pipeline)
    session.render()

    names = {schema["function"]["name"] for schema in session.tool_schemas}
    assert "characterize_user_understanding" in names
    assert "record_understanding" not in names


def test_tool_dispatches_via_event_pipeline_and_real_learning_driver():
    session, builder = _build_session()
    session.render()
    drivers = builder.build_drivers()
    learning_driver = next(d for d in drivers if isinstance(d, plugin.LearningDriver))
    call = {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "characterize_user_understanding",
            "arguments": json.dumps({
                "topic": "how WAL mode allows concurrent SQLite readers",
                "evidence": "explained WAL readers don't block writers",
                "understands": True,
            }),
        },
    }
    with Runtime(session, drivers=drivers) as rt:
        rt.post(InputReceived(message={"role": "assistant", "content": "", "tool_calls": [call]}))
        while not session.state.last_tool_calls:
            assert rt.step_once(timeout=2.0)
        learning_driver.quiesce(policy="terminate")

    assert session.state.last_tool_calls[0].result == "Recorded."
    assert not session.state.last_tool_calls[0].error
    assert len(_rows("observations")) == 1


@pytest.mark.parametrize("study", [False, True])
def test_study_flag_selects_the_visible_tools(study):
    state = State()
    state.durable(plugin.LEARNING_DURABLE_SPEC).study_active = study

    assert plugin.characterize_user_understanding.available(state) is (not study)
    for study_tool in plugin.STUDY_TOOLS:
        assert study_tool.available(state) is study


def test_study_tools_are_hidden_without_the_flag():
    assert not plugin.search_knowledge.available(State())
    assert plugin.characterize_user_understanding.available(State())


def test_study_command_wakes_the_agent_with_study_tools_and_a_first_step():
    session, _ = _build_session(with_interrupts=True, with_commands=True)
    sim, drv = _make_sim(session)
    try:
        session.render()

        def tool_names():
            return {schema["function"]["name"] for schema in session.tool_schemas}

        assert "characterize_user_understanding" in tool_names()
        assert "record_understanding" not in tool_names()
        assert not session.needs_reply

        sim.inject(SlashCommand("/study"))
        sim.run_until_idle()

        assert "record_understanding" in tool_names()
        assert "characterize_user_understanding" not in tool_names()
        content = session.state.entries[-1].messages[0]["content"]
        assert session.state.entries[-1].system_generated
        assert "ran /study" in content and "Step: orient" in content
        assert session.needs_reply
        assert "# Study session" in session.messages[0]["content"]

        already = plugin._study_on(SimpleNamespace(state=session.state))
        assert "already on" in already.outputs[0].text
        assert session.state.pending_interrupts == []

        sim.inject(SlashCommand("/study off"))
        sim.run_until_idle()
        assert session.state.durable("learning", plugin.LearningState).study_active is False
        assert "record_understanding" not in tool_names()
        assert "characterize_user_understanding" in tool_names()
    finally:
        drv.close()


def test_study_command_failure_reverts_study_active():
    session, _ = _build_session(with_interrupts=True, with_commands=True)
    effects = session.step(SlashCommand("/study"))
    assert session.state.durable("learning", plugin.LearningState).study_active is True
    eff = next(e for e in effects if isinstance(e, SubmitOp) and isinstance(e.request, plugin.StudyPickTopicOp))
    session.step(IoFailed(op_id=eff.op_id, error=RuntimeError("db locked")))
    assert session.state.durable("learning", plugin.LearningState).study_active is False


def test_study_instructions_render_only_during_study():
    session, _ = _build_session()

    assert session.render()[0]["content"] == "Base system"
    session.state.durable("learning", plugin.LearningState).study_active = True
    assert session.render()[0]["content"] == "Base system" + plugin.STUDY_PROMPT
    assert session.state.entries[0].messages[0]["content"] == "Base system"


def test_study_command_parses_an_optional_message():
    session, _ = _build_session()
    commands = [c for c in session.pipeline if isinstance(c, Command)]
    registry = CommandRegistry(commands)

    with_message = registry.parse("/study I'd like to continue learning about OFDM")
    assert with_message.command.path == "/study"
    assert with_message.arguments == {"message": "I'd like to continue learning about OFDM"}
    assert registry.parse("/study").arguments == {"message": ""}
    assert registry.parse("/study off").command.path == "/study off"


def test_study_message_follows_the_harness_message_and_is_not_an_answer():
    session, _ = _build_session(with_interrupts=True, with_commands=True)
    sim, drv = _make_sim(session)
    try:
        session.render()
        node = _add("OFDM subcarriers are spaced 1/T apart, which makes them orthogonal.")

        sim.inject(SlashCommand("/study I'd like to continue learning about OFDM"))
        sim.run_until_idle()

        harness, user = session.state.entries[-2:]
        assert harness.system_generated and "ran /study" in harness.messages[0]["content"]
        assert "message below says what they want to study" in harness.messages[0]["content"]
        assert "Possible holes" not in harness.messages[0]["content"]
        assert not user.system_generated
        assert user.messages[0]["content"] == (
            "--- USER ---\nI'd like to continue learning about OFDM\n--- END USER ---"
        )
        assert session.state.pending_interrupts == []
        assert session.needs_reply
        assert "record_understanding" in {s["function"]["name"] for s in session.tool_schemas}

        # Evidence drawn from the /study message itself is not graded as an answer.
        assert _record(session.state, node, True) == "Recorded."
        session.state.entries.append(_entry(len(session.state.entries), "user", "answer"))
        assert _record(session.state, node, False).startswith("Recorded. Next:\nStep: teach")
    finally:
        drv.close()


def test_study_message_while_already_studying_is_sent_as_a_user_message():
    session, _ = _build_session(with_commands=True)
    session.state.durable("learning", plugin.LearningState).study_active = True

    result = plugin._study_on(SimpleNamespace(state=session.state), message="switch to OFDM")

    assert "sent your message" in result.outputs[0].text
    assert session.state.entries[-1].messages[0]["content"].count("switch to OFDM") == 1


def test_prompts_scope_knowledge_to_the_wider_world_and_mention_rendering():
    observe = plugin.characterize_user_understanding.schema["function"]["description"]

    assert "wider world" in observe and "on this machine" in observe
    assert "wider world" in plugin.STUDY_PROMPT
    assert "LaTeX" in plugin.STUDY_PROMPT and "```mermaid" in plugin.STUDY_PROMPT


# --- config and storage ------------------------------------------------------


def test_config_defaults_without_a_file():
    cfg = plugin.load_config()

    assert cfg["cycle"] == plugin.DEFAULT_CONFIG["cycle"]
    assert cfg["_warnings"] == []


def test_config_file_overrides_and_reports_bad_values(tmp_path):
    _write_config(
        tmp_path,
        "cycle:\n  probe_questions: 5\n  review_questions: -1\n"
        "review:\n  growth_factor: 2\n  bogus: 1\n",
    )

    cfg = plugin.load_config()

    assert cfg["cycle"]["probe_questions"] == 5
    assert cfg["cycle"]["review_questions"] == plugin.DEFAULT_CONFIG["cycle"]["review_questions"]
    assert cfg["review"]["growth_factor"] == 2.0
    assert len(cfg["_warnings"]) == 2


def test_shipped_config_matches_defaults():
    shipped = yaml.safe_load((Path(__file__).parent.parent / "config" / "learning.yaml").read_text())

    assert shipped == plugin.DEFAULT_CONFIG


def test_connect_migrates_evidence_table_without_question_phase():
    path = plugin.db_path()
    path.parent.mkdir(parents=True)
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE evidence (id INTEGER PRIMARY KEY AUTOINCREMENT, node_id INTEGER NOT NULL,"
        " understands INTEGER NOT NULL, evidence TEXT NOT NULL, observed_at TEXT NOT NULL,"
        " source_kind TEXT NOT NULL, observation_id INTEGER, project TEXT NOT NULL DEFAULT '',"
        " session_id TEXT NOT NULL DEFAULT '', user_entry_index INTEGER)"
    )
    old.execute(
        "INSERT INTO evidence (node_id, understands, evidence, observed_at, source_kind)"
        " VALUES (1, 1, 'kept', '2026-11-01T00:00:00+00:00', 'study')"
    )
    old.commit()
    old.close()

    [row] = _rows("evidence")

    assert row["evidence"] == "kept"
    assert row["question_phase"] is None


def _settings_harness():
    session, _ = _build_session(with_settings=True)
    assert_pipeline_sans_io(session.pipeline)

    def dispatch(*, action="list", fail_persist=None, **kwargs):
        ev = SettingsRequest(request_id="r1", action=action, **kwargs)
        effects = session.step(ev)
        reply = None

        def _collect(effs):
            nonlocal reply
            next_events = []
            for eff in effs:
                if not isinstance(eff, SubmitOp):
                    continue
                req = eff.request
                if isinstance(req, ReplyFrontendRequest):
                    reply = req.payload
                elif isinstance(req, StoreUpdate):
                    if fail_persist is not None:
                        next_events.append(IoFailed(op_id=eff.op_id, error=fail_persist))
                    else:
                        drv = PluginStoreDriver()
                        inbox: queue.Queue = queue.Queue()
                        drv.start(inbox.put)
                        try:
                            drv.submit(eff.op_id, req)
                            io_ev = inbox.get(timeout=5.0)
                        finally:
                            drv.close()
                        next_events.append(io_ev)
            return next_events

        pending = _collect(effects)
        while pending:
            ev2 = pending.pop(0)
            pending.extend(_collect(session.step(ev2)))
        assert reply is not None
        return reply

    def catalog():
        res = dispatch(action="list")
        return {row["key"]: row for row in res["settings"] if row["key"].startswith("learning.")}

    def set_(key, value, *, fail_persist=None):
        listing = dispatch(action="list")
        return dispatch(
            action="set",
            binding=listing["binding"],
            key=key,
            value=value,
            fail_persist=fail_persist,
        )

    return catalog, set_


def test_settings_expose_every_config_option_with_defaults():
    catalog, _ = _settings_harness()
    rows = catalog()

    assert len(rows) == len(plugin.CONFIG_OPTIONS)
    probe = rows["learning.cycle.probe_questions"]
    assert (probe["kind"], probe["value"], probe["min"], probe["group"]) == (
        "int", 3, 1, "Learning / Study cycle"
    )
    low = rows["learning.accuracy.low"]
    assert (low["kind"], low["value"], low["apply"]) == ("text", "0.75", "live")
    assert all(row["error"] is None and row["writable"] for row in rows.values())


def test_settings_write_the_config_file_via_store_update(tmp_path):
    shipped = (Path(__file__).parent.parent / "config" / "learning.yaml").read_text()
    _write_config(tmp_path, shipped)
    catalog, set_ = _settings_harness()

    assert set_("learning.cycle.probe_questions", 5)["ok"]
    assert set_("learning.accuracy.low", " 0.6 ")["ok"]

    cfg = plugin.load_config()
    assert cfg["cycle"]["probe_questions"] == 5 and cfg["accuracy"]["low"] == 0.6
    assert cfg["_warnings"] == []
    assert catalog()["learning.accuracy.low"]["value"] == "0.6"


def test_settings_create_the_config_file_when_missing():
    _, set_ = _settings_harness()

    assert set_("learning.scoring.half_life_days", "14")["ok"]

    assert plugin.load_config()["scoring"]["half_life_days"] == 14.0
    assert yaml.safe_load(plugin.config_path().read_text()) == {"scoring": {"half_life_days": 14.0}}


def test_settings_rollback_in_memory_on_persist_failure():
    catalog, set_ = _settings_harness()
    res = set_(
        "learning.cycle.probe_questions",
        9,
        fail_persist=PermissionError("disk full"),
    )
    assert not res["ok"]
    assert catalog()["learning.cycle.probe_questions"]["value"] == 3
    assert not plugin.config_path().exists()


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("learning.accuracy.high", "lots", "must be a number"),
        ("learning.accuracy.high", "1.5", "at most 1"),
        ("learning.review.growth_factor", "-2", "at least 0"),
        ("learning.review.growth_factor", "nan", "finite"),
    ],
)
def test_settings_reject_invalid_values_without_writing(key, value, message):
    _, set_ = _settings_harness()

    result = set_(key, value)

    assert not result["ok"] and message in result["error"]["message"]
    assert not plugin.config_path().exists()


def test_settings_refuse_to_edit_an_unparseable_file(tmp_path):
    _write_config(tmp_path, "cycle: [unclosed\n")
    _, set_ = _settings_harness()

    result = set_("learning.cycle.candidates", 3)

    assert not result["ok"] and "not a valid YAML mapping" in result["error"]["message"]
    assert plugin.config_path().read_text() == "cycle: [unclosed\n"


def test_setting_a_value_clears_its_own_warning(tmp_path):
    _write_config(tmp_path, "cycle:\n  candidates: -3\n")
    _, set_ = _settings_harness()

    assert set_("learning.cycle.candidates", 4)["ok"]
    cfg = plugin.load_config()
    assert cfg["cycle"]["candidates"] == 4 and cfg["_warnings"] == []


# --- passive observations ----------------------------------------------------


def test_records_observation_with_pointer_to_latest_user_entry():
    state = _session_state(
        _entry(0, "system", "sys"),
        _entry(1, "user", "why don't my threads speed this up?"),
        _entry(2, "assistant", "..."),
        _entry(3, "user", "[interrupt]", system_generated=True),
    )

    result = _run_tool_gen(
        plugin.characterize_user_understanding,
        state,
        topic="why the GIL stops CPU-bound Python threads from running in parallel",
        evidence="Expected threading to speed up a CPU-bound loop.",
        understands=False,
    )

    assert result == "Recorded."
    [row] = _rows("observations")
    assert row["understands"] == 0
    assert (row["project"], row["session_id"], row["user_entry_index"]) == ("demo", "session-1", 1)


def test_records_observation_without_session_state():
    result = _run_tool_gen(
        plugin.characterize_user_understanding,
        None,
        topic="topic",
        evidence="evidence",
        understands=True,
    )

    assert result == "Recorded."
    [row] = _rows("observations")
    assert row["session_id"] == "" and row["user_entry_index"] is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"topic": " ", "evidence": "e", "understands": True}, "topic"),
        ({"topic": "t", "evidence": "", "understands": True}, "evidence"),
        ({"topic": "t", "evidence": "e", "understands": "false"}, "understands"),
    ],
)
def test_rejects_invalid_observations(kwargs, message):
    result = _run_tool_gen(plugin.characterize_user_understanding, None, **kwargs)

    assert result.startswith("ERROR") and message in result
    assert _rows("observations") == []


def test_observation_attach_and_dismiss():
    for topic, understands in (("lifetimes", False), ("borrowing", True), ("macros", False)):
        _run_tool_gen(
            plugin.characterize_user_understanding,
            None,
            topic=topic,
            evidence=f"said something about {topic}",
            understands=understands,
        )
    node = _add("A reference cannot outlive the value it borrows.")

    assert _run_tool_gen(plugin.attach_observation, None, observation_id=1, node_id=node).startswith("Attached")
    assert "already attached" in _run_tool_gen(plugin.attach_observation, None, observation_id=1, node_id=node)
    assert _run_tool_gen(plugin.dismiss_observation, None, observation_id=3).startswith("Dismissed")

    _, step = _start()
    assert "observation 2" in step
    assert "observation 1" not in step and "observation 3" not in step
    [evidence] = _rows("evidence")
    [observation] = [row for row in _rows("observations") if row["id"] == 1]
    assert evidence["source_kind"] == "observation"
    assert evidence["understands"] == 0
    assert evidence["observed_at"] == observation["created_at"]


# --- graph editing -----------------------------------------------------------


def test_add_search_and_view_knowledge():
    base = _add("CPython's GIL lets only one thread execute Python bytecode at a time.")
    top = _add(
        "CPU-bound Python threads do not run in parallel under the GIL.", depends_on=[base]
    )
    state = _session_state(_entry(0, "user", "threads share one GIL, right?"), study=True)

    assert _record(state, top, True).startswith("Recorded.")
    found = _run_tool_gen(plugin.search_knowledge, None, query="threads in parallel")
    assert f"node {top}:" in found
    view = _run_tool_gen(plugin.view_knowledge, None, node_id=base)
    assert "Depended on by:" in view and f"node {top}:" in view
    assert "possible hole" in view
    view = _run_tool_gen(plugin.view_knowledge, None, node_id=top)
    assert "understands (study; project demo, session session-1, entry 0" in view
    assert "review interval 1 days" in view


def test_graph_edits_validate_ids_and_cycles():
    a = _add("A")
    b = _add("B", depends_on=[a])

    assert _run_tool_gen(plugin.add_knowledge, None, statement="C", depends_on=[99]).startswith("ERROR")
    assert _run_tool_gen(plugin.add_dependency, None, node_id=a, depends_on_id=a).startswith("ERROR")
    assert "cycle" in _run_tool_gen(plugin.add_dependency, None, node_id=a, depends_on_id=b)
    assert _run_tool_gen(
        plugin.record_understanding, None, node_id=99, understands=True, evidence="e"
    ).startswith("ERROR")
    c = _add("C")
    assert _run_tool_gen(plugin.add_dependency, None, node_id=c, depends_on_id=b).startswith("Node")
    assert _states()[b].dependents == (c,)


# --- derived state -----------------------------------------------------------


def test_evidence_decays_with_age_and_observations_count_less():
    fresh, old, passive = _add("fresh"), _add("old"), _add("passive")
    _evidence(fresh, True)
    _evidence(old, True, days_ago=plugin.DEFAULT_CONFIG["scoring"]["half_life_days"])
    _evidence(passive, True, kind="observation")

    states = _states()
    assert states[fresh].net == pytest.approx(1.0, abs=1e-3)
    assert states[old].net == pytest.approx(0.5, abs=1e-3)
    assert states[passive].net == pytest.approx(
        plugin.DEFAULT_CONFIG["scoring"]["observation_weight"], abs=1e-3
    )


def test_untested_foundation_under_known_node_is_a_hole():
    foundation = _add("foundation")
    tested_foundation = _add("tested foundation")
    known = _add("known", depends_on=[foundation, tested_foundation])
    _evidence(known, True)
    _evidence(tested_foundation, True)

    states = _states()
    assert states[foundation].hole_score > states[tested_foundation].hole_score
    _, step = _start()
    assert "Possible holes and edges across the graph" in step
    assert step.index(f"node {foundation}:") < step.index(f"node {tested_foundation}:")


def test_failed_node_on_solid_foundation_is_an_edge():
    solid = _add("solid")
    failed = _add("failed", depends_on=[solid])
    shaky = _add("shaky")
    failed_on_shaky = _add("failed on shaky", depends_on=[shaky])
    _evidence(solid, True)
    _evidence(failed, False)
    _evidence(shaky, False)
    _evidence(failed_on_shaky, False)

    states = _states()
    assert states[failed].edge_score > states[failed_on_shaky].edge_score
    assert states[solid].edge_score == 0
    _, step = _start()
    assert f"possible edge [{states[failed].edge_score:.2f}]; node {failed}:" in step


def test_review_interval_grows_with_spaced_successes_and_resets_on_a_miss():
    once, spaced, crammed, lapsed = _add("once"), _add("spaced"), _add("crammed"), _add("lapsed")
    _evidence(once, True, days_ago=2)
    _evidence(spaced, True, days_ago=10)
    _evidence(spaced, True, days_ago=8)
    _evidence(spaced, True, days_ago=1)
    _evidence(crammed, True, hours_ago=3)
    _evidence(crammed, True, hours_ago=2)
    _evidence(crammed, True, hours_ago=1)
    _evidence(lapsed, True, days_ago=20)
    _evidence(lapsed, True, days_ago=10)
    _evidence(lapsed, False, days_ago=2)

    states = _states()
    assert states[once].interval_days == 1.0
    assert states[once].urgency == pytest.approx(2.0, rel=1e-3)
    assert states[spaced].spaced_successes == 3
    assert states[spaced].interval_days == 9.0
    assert states[crammed].spaced_successes == 1
    assert states[lapsed].spaced_successes == 0
    assert states[lapsed].interval_days == 1.0


def test_due_nodes_are_offered_by_priority_not_queued():
    for days in (1.5, 3, 6):
        _evidence(_add(f"seen {days} days ago"), True, days_ago=days)
    _evidence(_add("not due"), True, hours_ago=2)

    _, step = _start()

    assert step.startswith("Step: review (question 1 of up to 2")
    assert step.index("seen 6 days ago") < step.index("seen 3 days ago") < step.index("seen 1.5")
    assert "not due" not in step


# --- the study cycle ---------------------------------------------------------


def test_empty_graph_orients_then_the_first_answer_drives_the_cycle():
    state, step = _start()
    assert step.startswith("Step: orient")

    state.entries.append(_entry(len(state.entries), "user", "I want to learn Mamba-3"))
    node = _add("Mamba-3 uses an exponential-trapezoidal discretization.")
    result = _answer(state, node, False)

    assert result.startswith("Recorded. Next:\nStep: teach")
    assert f"missed node {node}" in result


def test_only_the_first_record_for_each_user_message_advances_the_cycle():
    target, other = _add("target"), _add("other")
    state, _ = _start()

    assert _answer(state, target, False).startswith("Recorded. Next:")
    assert _record(state, other, True) == "Recorded."
    phases = {row["node_id"]: row["question_phase"] for row in _rows("evidence")}
    assert phases == {target: "probe", other: None}


def test_messages_from_before_study_are_not_answers():
    node = _add("node")
    state = State(entries=[_entry(0, "system", "sys"), _entry(1, "user", "earlier chat")])
    _start(state)

    assert _record(state, node, True) == "Recorded."
    assert _answer(state, node, True).startswith("Recorded. Next:")


def test_records_outside_study_never_advance():
    node = _add("node")
    state = State(entries=[_entry(0, "user", "hi")])

    assert _record(state, node, True) == "Recorded."


def test_probe_miss_is_taught_with_its_foundations_and_check_ends_the_cycle():
    base = _add("base")
    node = _add("node", depends_on=[base])
    _evidence(base, True, hours_ago=1)
    state, _ = _start()

    teach = _answer(state, node, False)
    assert "Step: teach" in teach and f"missed node {node}: node" in teach
    assert "It rests on:" in teach and f"node {base}: base" in teach
    assert "worked example" in teach

    after = _answer(state, node, True)
    assert "recorded as correct" in after
    cycle = state.durable("learning", plugin.LearningState).cycle
    assert "Step: probe" in after and cycle["probes_done"] == 0
    assert f"Nodes near node {node}" in after and f"node {base}: base" in after


def test_correct_probes_show_what_builds_on_them_then_teach_beyond():
    a = _add("a")
    b = _add("b", depends_on=[a])
    c = _add("c", depends_on=[b])
    state, _ = _start()

    step = _answer(state, a, True)
    assert f"answered node {a} correctly" in step and f"node {b}: b" in step
    step = _answer(state, b, True)
    assert f"answered node {b} correctly" in step and f"node {c}: c" in step
    teach = _answer(state, c, True)

    assert "Step: teach" in teach and f"ending at node {c}" in teach


def test_the_agent_may_answer_on_a_different_node_than_suggested():
    a = _add("a")
    unrelated = _add("unrelated")
    state, _ = _start()

    step = _answer(state, unrelated, True)
    cycle = state.durable("learning", plugin.LearningState).cycle

    assert f"answered node {unrelated} correctly" in step
    assert cycle["anchor_node_id"] == unrelated
    assert a not in cycle["asked"]


def test_reteaches_then_moves_on_after_max_attempts():
    node = _add("hard")
    state, _ = _start()
    _answer(state, node, False)

    assert "different angle" in _answer(state, node, False)
    after = _answer(state, node, False)

    assert "still hasn't landed" in after
    assert "Step: teach" not in after


def test_cycle_starts_with_due_reviews_then_probes():
    due = _add("due")
    fresh = _add("fresh")
    _evidence(due, True, days_ago=3)
    _evidence(fresh, False)
    state, review = _start()

    assert review.startswith("Step: review")
    assert f"node {due}: due" in review
    after = _answer(state, due, True)

    assert "recorded as correct" in after and "Step: probe" in after


def test_review_miss_is_corrected_and_rechecked():
    node = _add("forgotten")
    _evidence(node, True, days_ago=5)
    state, _ = _start()

    teach = _answer(state, node, False)

    assert f"missed node {node} in review" in teach
    assert state.durable("learning", plugin.LearningState).cycle["phase"] == "teach"


def test_low_accuracy_limits_probing_to_one_question():
    filler = _add("filler")
    for _ in range(5):
        _evidence(filler, False, hours_ago=1, phase="probe")
    node = _add("node")
    state, step = _start()

    assert "of up to 1 before teaching" in step
    assert "below the 75% target" in step
    assert "Step: teach" in _answer(state, node, True)


def test_probe_offers_untested_observations():
    _run_tool_gen(
        plugin.characterize_user_understanding,
        None,
        topic="selective scan",
        evidence="asked what it was",
        understands=False,
    )

    _, step = _start()

    assert "observation 1" in step and "attach_observation" in step


def test_config_changes_cycle_lengths(tmp_path):
    _write_config(tmp_path, "cycle:\n  probe_questions: 1\n")
    node = _add("node")
    state, _ = _start()

    assert "Step: teach" in _answer(state, node, True)
