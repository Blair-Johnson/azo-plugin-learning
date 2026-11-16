# azo-plugin-learning

An Agent Zoo userspace plugin that builds a persistent model of what the user
understands, while they work and while they study between agent turns.

## Passive observations

In normal sessions, agents see one tool:
`characterize_user_understanding(topic, evidence, understands)`. They call it
whenever the user reveals that they do or don't understand something about the
wider world, such as how a technology, algorithm, model, or mathematical or
scientific idea works, even when it is unrelated to the current task. Details
of local code and projects are out of scope. Each call appends an observation
to a global queue, together with the project, session id, and index of the
user's latest message, so a study session can revisit the transcript.

## Study sessions

`/study` turns the current session into a study session and wakes the agent
with the study tools, the study instructions, and a first step. `/study off`
turns it back.

The plugin keeps the session's rhythm, and the agent decides the content. After
each answer, the agent records evidence with `record_understanding`; the first
record for each user message is graded as the answer, and its result is the next
step: review, probe, or teach, with the graph context behind it. That context
lists candidate nodes with their evidence and hole or edge flags, nodes due for
review, and untested observations. The agent reads them, explores the graph,
and chooses what to ask or teach from the graph and the user's answers.

Each cycle has three parts:

1. **Review.** Up to two questions on nodes that are due.
2. **Probe.** Up to three questions, each a step harder than the last correct
   answer. A miss ends probing immediately, because a failed attempt helps
   learning only when the answer follows soon after.
3. **Teach.** A short explanation of what the user missed, or of the next step
   beyond their last success, ending in a check question. A missed check is
   retaught once from a different angle; then the node is left to review.

If recent accuracy falls below 75%, cycles probe only once and the agent is
told to ask easier questions; above 90%, harder ones. Probe candidates come
from around the node the user last answered, so the session's focus follows
the conversation. The study prompt tells the agent that the terminal renders
Markdown, LaTeX, and Mermaid.

| Tool | Purpose |
| --- | --- |
| `record_understanding` | Record evidence after an answer; returns the next step |
| `view_knowledge` | Show a node, its neighbours, evidence, and review timing |
| `search_knowledge` | Find existing nodes before adding one |
| `add_knowledge` | Add a true statement, with its dependencies |
| `add_dependency` | Link a node to one it depends on |
| `attach_observation` | Turn an observation into evidence on a node |
| `dismiss_observation` | Drop an observation not worth studying |

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

The same settings appear in the TUI under `/config` → Plugins → Learning.
Changes there are validated and written to that file, keeping its comments.
Decimal settings are edited as text, because plugin settings have no decimal
type. If the file isn't simple block-style YAML that can be edited safely, the
change is refused and the file is left untouched.

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
