# .claude/.agent/ — passive internal context

This directory is the molcfg project's **passive** agent zone. Content here
outlives any single feature: evolving notes, architectural decisions,
open questions, debt log, and the project blueprint.

It is **not** for:

- public-user documentation → `docs/`
- active in-flight specs → `.claude/specs/` (they are deleted on completion)
- Claude Code runtime config → `.claude/agents/`, `.claude/skills/`, etc.

## Files

| File | Purpose | Maintained by |
|---|---|---|
| `notes.md` | Evolving architectural decisions, captured rules, invariants | `/mol:note` |
| `architecture.md` | Project blueprint — module catalog, public surface, layer roles | `/mol:map` |
| `open-questions.md` | Unresolved questions recorded during bootstrap or later | manual / agents |

Add `contracts/`, `rubrics/`, `decisions/`, `debt/`, or `handoffs/`
subdirectories **only when you have real content for them**. Empty
scaffolding is not value.
