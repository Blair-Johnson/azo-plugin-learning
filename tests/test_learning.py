from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from agent_utils import PipelineBuilder, Session, core, interrupts
from agent_utils.types import Entry

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


def _session_state(tmp_path, *entries, study=False):
    state = SimpleNamespace(
        run_db=SimpleNamespace(
            session_id="session-1",
            agent_db=SimpleNamespace(path=str(tmp_path / "projects" / "demo" / "db.sqlite")),
        ),
        entries=list(entries),
    )
    setattr(state, plugin.STUDY_FLAG, study)
    return state


def _build_session():
    session = Session(system_prompt="Base system")
    builder = PipelineBuilder(session).add(core())
    plugin.register_features(builder, session=session, config=SimpleNamespace())
    session.pipeline = builder.build()
    return session


def _add(statement, depends_on=None):
    result = plugin.add_knowledge.fn(statement=statement, depends_on=depends_on)
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
    path = plugin.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _start():
    """Run /study on a bare session state and return (state, first step)."""

    state = SimpleNamespace(entries=[_entry(0, "system", "sys")], pending_interrupts=[])
    plugin._study_on(SimpleNamespace(state=state, session=None))
    message = state.pending_interrupts[-1]
    assert message.startswith(plugin.STUDY_START_MESSAGE)
    return state, message.removeprefix(plugin.STUDY_START_MESSAGE)


def _record(state, node, understands):
    return plugin.record_understanding.fn(
        node_id=node, understands=understands, evidence="answer", state=state
    )


def _answer(state, node, understands):
    """The user sends a message, and the agent records it on ``node``."""

    state.entries.append(_entry(len(state.entries), "user", "answer"))
    return _record(state, node, understands)


# --- pipeline, visibility, commands, prompt ---------------------------------

def test_pipeline_builds_with_all_tools():
    session = _build_session()

    assert plugin.characterize_user_understanding in session.pipeline
    for study_tool in plugin.STUDY_TOOLS:
        assert study_tool in session.pipeline


@pytest.mark.parametrize("study", [False, True])
def test_study_flag_selects_the_visible_tools(study):
    state = SimpleNamespace(**{plugin.STUDY_FLAG: study})

    assert plugin.characterize_user_understanding.available(state) is (not study)
    for study_tool in plugin.STUDY_TOOLS:
        assert study_tool.available(state) is study


def test_study_tools_are_hidden_without_the_flag():
    assert not plugin.search_knowledge.available(SimpleNamespace())
    assert plugin.characterize_user_understanding.available(SimpleNamespace())


def test_study_command_wakes_the_agent_with_study_tools_and_a_first_step():
    session = Session(system_prompt="Base system")
    builder = PipelineBuilder(session).add(core()).add(interrupts())
    plugin.register_features(builder, session=session, config=SimpleNamespace())
    session.pipeline = builder.build()
    session.run()
    ctx = SimpleNamespace(state=session.state, session=session)

    def tool_names():
        return {schema["function"]["name"] for schema in session.tool_schemas}

    assert "characterize_user_understanding" in tool_names()
    assert "record_understanding" not in tool_names()
    assert not session.needs_reply

    plugin._study_on(ctx)

    # The idle runtime delivers interrupts without a full pipeline pass, so the
    # command itself must have refreshed the published tools.
    assert session.poll_interrupts()
    assert "record_understanding" in tool_names()
    assert "characterize_user_understanding" not in tool_names()
    content = session.state.entries[-1].messages[0]["content"]
    assert session.state.entries[-1].system_generated
    assert "ran /study" in content and "Step: orient" in content
    assert session.needs_reply
    assert "# Study session" in session.messages[0]["content"]

    assert "already on" in plugin._study_on(ctx).outputs[0].text
    assert session.state.pending_interrupts == []

    plugin._study_off(ctx)
    assert getattr(session.state, plugin.STUDY_FLAG) is False
    assert "record_understanding" not in tool_names()
    assert "characterize_user_understanding" in tool_names()


def test_study_instructions_render_only_during_study():
    session = _build_session()

    assert session.rerender()[0]["content"] == "Base system"
    setattr(session.state, plugin.STUDY_FLAG, True)
    assert session.rerender()[0]["content"] == "Base system" + plugin.STUDY_PROMPT
    assert session.state.entries[0].messages[0]["content"] == "Base system"


def test_study_command_parses_an_optional_message():
    from agent_zoo.runtime.commands import CommandRegistry

    session = _build_session()
    commands = [c for c in session.pipeline if type(c).__name__ == "Command"]
    registry = CommandRegistry(commands)

    with_message = registry.parse("/study I'd like to continue learning about OFDM")
    assert with_message.command.path == "/study"
    assert with_message.arguments == {"message": "I'd like to continue learning about OFDM"}
    assert registry.parse("/study").arguments == {"message": ""}
    assert registry.parse("/study off").command.path == "/study off"


def test_study_message_follows_the_harness_message_and_is_not_an_answer():
    session = Session(system_prompt="Base system")
    builder = PipelineBuilder(session).add(core()).add(interrupts())
    plugin.register_features(builder, session=session, config=SimpleNamespace())
    session.pipeline = builder.build()
    session.run()
    node = _add("OFDM subcarriers are spaced 1/T apart, which makes them orthogonal.")

    plugin._study_on(
        SimpleNamespace(state=session.state, session=session),
        message="I'd like to continue learning about OFDM",
    )

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


def test_study_message_while_already_studying_is_sent_as_a_user_message():
    session = _build_session()
    ctx = SimpleNamespace(state=session.state, session=session)
    setattr(session.state, plugin.STUDY_FLAG, True)

    result = plugin._study_on(ctx, message="switch to OFDM")

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
    import yaml
    from pathlib import Path

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


# --- passive observations ----------------------------------------------------

def test_records_observation_with_pointer_to_latest_user_entry(tmp_path):
    state = _session_state(
        tmp_path,
        _entry(0, "system", "sys"),
        _entry(1, "user", "why don't my threads speed this up?"),
        _entry(2, "assistant", "..."),
        _entry(3, "user", "[interrupt]", system_generated=True),
    )

    result = plugin.characterize_user_understanding.fn(
        topic="why the GIL stops CPU-bound Python threads from running in parallel",
        evidence="Expected threading to speed up a CPU-bound loop.",
        understands=False,
        state=state,
    )

    assert result == "Recorded."
    [row] = _rows("observations")
    assert row["understands"] == 0
    assert (row["project"], row["session_id"], row["user_entry_index"]) == ("demo", "session-1", 1)


def test_records_observation_without_session_state():
    result = plugin.characterize_user_understanding.fn(
        topic="topic", evidence="evidence", understands=True
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
    result = plugin.characterize_user_understanding.fn(**kwargs)

    assert result.startswith("ERROR") and message in result
    assert _rows("observations") == []


def test_observation_attach_and_dismiss():
    for topic, understands in (("lifetimes", False), ("borrowing", True), ("macros", False)):
        plugin.characterize_user_understanding.fn(
            topic=topic, evidence=f"said something about {topic}", understands=understands
        )
    node = _add("A reference cannot outlive the value it borrows.")

    assert plugin.attach_observation.fn(observation_id=1, node_id=node).startswith("Attached")
    assert "already attached" in plugin.attach_observation.fn(observation_id=1, node_id=node)
    assert plugin.dismiss_observation.fn(observation_id=3).startswith("Dismissed")

    _, step = _start()
    assert "observation 2" in step
    assert "observation 1" not in step and "observation 3" not in step
    [evidence] = _rows("evidence")
    [observation] = [row for row in _rows("observations") if row["id"] == 1]
    assert evidence["source_kind"] == "observation"
    assert evidence["understands"] == 0
    assert evidence["observed_at"] == observation["created_at"]


# --- graph editing -----------------------------------------------------------

def test_add_search_and_view_knowledge(tmp_path):
    base = _add("CPython's GIL lets only one thread execute Python bytecode at a time.")
    top = _add(
        "CPU-bound Python threads do not run in parallel under the GIL.", depends_on=[base]
    )
    state = _session_state(tmp_path, _entry(0, "user", "threads share one GIL, right?"), study=True)

    assert _record(state, top, True).startswith("Recorded.")
    found = plugin.search_knowledge.fn(query="threads in parallel")
    assert f"node {top}:" in found
    view = plugin.view_knowledge.fn(node_id=base)
    assert "Depended on by:" in view and f"node {top}:" in view
    assert "possible hole" in view
    view = plugin.view_knowledge.fn(node_id=top)
    assert "understands (study; project demo, session session-1, entry 0" in view
    assert "review interval 1 days" in view


def test_graph_edits_validate_ids_and_cycles():
    a = _add("A")
    b = _add("B", depends_on=[a])

    assert plugin.add_knowledge.fn(statement="C", depends_on=[99]).startswith("ERROR")
    assert plugin.add_dependency.fn(node_id=a, depends_on_id=a).startswith("ERROR")
    assert "cycle" in plugin.add_dependency.fn(node_id=a, depends_on_id=b)
    assert plugin.record_understanding.fn(node_id=99, understands=True, evidence="e").startswith(
        "ERROR"
    )
    c = _add("C")
    assert plugin.add_dependency.fn(node_id=c, depends_on_id=b).startswith("Node")
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
    state = SimpleNamespace(
        entries=[_entry(0, "system", "sys"), _entry(1, "user", "earlier chat")],
        pending_interrupts=[],
    )
    plugin._study_on(SimpleNamespace(state=state, session=None))

    assert _record(state, node, True) == "Recorded."
    assert _answer(state, node, True).startswith("Recorded. Next:")


def test_records_outside_study_never_advance():
    node = _add("node")
    state = SimpleNamespace(entries=[_entry(0, "user", "hi")])

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
    assert "Step: probe" in after and state.learning_study_cycle["probes_done"] == 0
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

    assert f"answered node {unrelated} correctly" in step
    assert state.learning_study_cycle["anchor_node_id"] == unrelated
    assert a not in state.learning_study_cycle["asked"]


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
    assert state.learning_study_cycle["phase"] == "teach"


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
    plugin.characterize_user_understanding.fn(
        topic="selective scan", evidence="asked what it was", understands=False
    )

    _, step = _start()

    assert "observation 1" in step and "attach_observation" in step


def test_config_changes_cycle_lengths(tmp_path):
    _write_config(tmp_path, "cycle:\n  probe_questions: 1\n")
    node = _add("node")
    state, _ = _start()

    assert "Step: teach" in _answer(state, node, True)
