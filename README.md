# azo-plugin-learning

An Agent Zoo userspace plugin that builds a persistent model of what the user
understands, while they work and while they study between agent turns.

## Passive observations

In normal sessions, agents see one tool:
`characterize_user_understanding(topic, evidence, understands)`. They call it
whenever the user reveals that they do or don't understand something, even when
it is unrelated to the current task. Each call appends an observation to a
global queue, together with the project, session id, and index of the user's
latest message, so a study session can revisit the transcript.

## Study sessions

`/study` turns the current session into a study session, and `/study off` turns
it back. The plugin runs the session; the agent calls `next_study_step` after
each answer and does what it says. Each cycle has three parts:

1. **Review.** Up to two questions on nodes that are due, ranked by priority.
2. **Probe.** Up to three questions that climb from the focus toward harder
   nodes. A miss ends probing immediately, because a failed attempt helps
   learning only when the answer follows soon after.
3. **Teach.** A short explanation of the missed node, or of the next step
   beyond the last success, ending in a check question. A missed check is
   retaught once from a different angle; then the node is left to review.

Teaching uses a worked example when the material is new to the user, and a
guided problem when its foundations are solid. If recent accuracy falls below
75%, cycles probe only once and pick easier targets; above 90%, they aim
harder. A long gap since the last step adds a one-line recap.

| Tool | Purpose |
| --- | --- |
| `next_study_step` | Get the next step and its target node |
| `set_study_focus` | Focus probing on a node and its neighbourhood |
| `record_understanding` | Record evidence about a node after an answer |
| `search_knowledge` | Find existing nodes before adding one |
| `add_knowledge` | Add a true statement, with its dependencies |
| `add_dependency` | Link a node to one it depends on |
| `view_knowledge` | Show a node, its neighbours, evidence, and review timing |
| `find_study_targets` | Rank holes, edges, and nodes due for review |
| `list_observations` | Show queued observations not yet handled |
| `attach_observation` | Turn an observation into evidence on a node |
| `dismiss_observation` | Drop an observation the user doesn't care about |

## The graph is the only data structure

Nodes and dependency links are objective knowledge. Evidence about the user is
stored separately and never written into nodes. Nothing stores due dates or
scores: everything is computed from the evidence when it is read.

A node's review interval starts at one day and triples with each success that
is at least eight hours after the previous counted success. A miss resets it.
Its urgency is the time since its last evidence divided by that interval, and
it is due once urgency reaches 1. Due nodes are ranked by urgency, with a bonus
for nodes that others depend on, so review never builds up a backlog: the
session simply takes the most urgent nodes when it has time. Evidence recorded
by working agents counts as review too.

A **hole** is a node that well-understood nodes depend on, but whose own
evidence is thin, old, or negative. An **edge** is a node that failed or is
untested while its foundations look solid.

## Configuration

Every threshold above is configurable in
`<state-home>/plugin-configs/azo-plugin-learning/config/learning.yaml`. The
installer copies `config/learning.yaml` there if it is missing and preserves it
afterwards. Edits apply on the next tool call. Invalid values keep their
defaults and are reported in the next study step.

## Storage

Everything lives in `<state-home>/learning/learning.sqlite`, where
`<state-home>` is `AGENT_ZOO_HOME`, `AGENT_ZOO_STATE_ROOT`, or
`~/.local/share/agent-zoo`. It is shared by every project and session.

```bash
sqlite3 ~/.local/share/agent-zoo/learning/learning.sqlite \
  "SELECT id, created_at, understands, topic FROM observations ORDER BY id DESC LIMIT 20"
```

## Install and test

```bash
pixi run azo-plugin install ~/Documents/GitHub/azo-plugin-learning --force
```

Then run `/reload` in the AZO TUI. Run the tests with:

```bash
pixi run --environment test test
```
