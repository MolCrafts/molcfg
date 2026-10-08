# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If you want to work on the documentation site:

```bash
pip install -e ".[docs]"
```

## Local Checks

Run lint and tests before opening a pull request:

```bash
ruff check .
pytest -q
```

If documentation dependencies are installed, preview docs locally:

```bash
zensical serve
```

## CI

One workflow per kind of work; shared setup comes from
`MolCrafts/molcrafts-ci/actions/<name>@master`. Every workflow's first job,
`<file> / context`, runs `MolCrafts/molcrafts-ci/actions/ci-context@master`,
and the other jobs read its outputs. It picks the tier:
the *fast* tier runs on a feature-branch push to MolCrafts; the *full* tier on
every push to a fork (so a branch is proven before its pull request), on
`dev`/`master`/`main` on MolCrafts, on pull requests, tags and dispatches. A
pull request inside a fork is skipped (its push already ran the full tier).

| workflow | fast tier (feature branch on MolCrafts) | full tier (fork pushes, dev/master/main, PRs, tags) | MolCrafts only |
|---|---|---|---|
| `lint.yml` | `lint / hooks` (pre-commit stage, all files), `lint / workflows` (MolCrafts/molcrafts-ci/actions/check-workflows) | same | — |
| `test.yml` | `test / context`, `test / python (ubuntu-latest, 3.12)` | `test / context`, `test / python ({ubuntu,macos,windows}-latest, {3.12,3.14})` | — |
| `docs.yml` | `docs / build` (`zensical build --strict`) | same | deploy: Cloudflare Pages, outside Actions |
| `release.yml` | — | — | `v*` tag: lint + test + `release / build` + `release / pypi` (only a `v*` tag on MolCrafts: `publish`); `workflow_dispatch` = dry run (no upload) |

The `protect-master` ruleset on `master` requires a pull request, blocks force
pushes and deletion, and requires `test / context` and the full-tier `lint /`, `test /` and `docs /` checks.

## Change Expectations

- Keep the runtime package dependency free.
- Add tests for behavior changes and bug fixes.
- Preserve backwards compatibility for public APIs unless the change is explicitly scheduled for a major release.
- Update `README.md` and `docs/` when behavior or public interfaces change.
  Release history lives in git tags / GitHub Releases (no `CHANGELOG.md`).

## Pull Requests

- Keep PRs scoped to one logical change.
- Describe user-visible impact clearly.
- Include migration notes when behavior changes.
