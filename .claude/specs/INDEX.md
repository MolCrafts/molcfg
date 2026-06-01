# Specs INDEX

One-line entry per live spec. `/mol:spec` appends here; `/mol:impl` prunes the
entry when the spec's tasks are complete and the spec file is deleted.

<!-- /mol:spec appends entries below this line -->

- [interpolation-grammar](interpolation-grammar.md) — rewrite `${...}` engine to vscode+omegaconf grammar (env / absolute / relative / `, default`, dict+list+tuple recursion); split into `interpolation.py`; replace `CircularReferenceError` with `ParserError`/`InterpretError`; no backwards compat [approved]
