# Open questions

Questions raised during `/mol-agent:bootstrap` (2026-05-06) that the project
maintainer should answer or resolve as the harness gets used.

- **Janitor knobs**: should `style.max_function_lines` be tightened from the
  default 80? Several functions in `src/molcfg/config.py` and
  `src/molcfg/validation.py` approach but stay under the default — review
  after a few `/mol:simplify` cycles.
- **Debt window**: no `TODO`/`FIXME` markers exist in `src/` today. Default
  60-day window is fine; revisit only if markers start accumulating.
- **`ty` in CI**: pre-commit runs `ty check src/`, but `.github/workflows/ci.yml`
  does not. Intentional (pre-commit is the gate) or oversight? If
  intentional, document; if not, add to CI.
- **Coverage gate**: `pyproject.toml` has no coverage threshold and CI does
  not run `pytest --cov`. Decide whether to enforce a floor.
