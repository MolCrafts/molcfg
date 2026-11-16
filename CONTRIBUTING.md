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

Shared setup comes from MolCrafts/molcrafts-ci, every reference pinned to one
commit SHA. Each workflow's first job, `<file> / context`, calls
`MolCrafts/molcrafts-ci/.github/workflows/context.yml`, and every other job
gates on its outputs; `test / context` picks the tier: the *fast* tier runs on
a feature-branch push to MolCrafts; the *full* tier on every push to a fork (so
a branch is proven before its pull request), on `dev`/`master` on MolCrafts,
on pull requests, tags and dispatches. A pull request inside a fork is skipped
(its push already ran the full tier).

| workflow | fast tier (feature branch on MolCrafts) | full tier (fork pushes, dev/master, PRs, tags) | MolCrafts only |
|---|---|---|---|
| `test.yml` | `test / python (ubuntu-latest, 3.12)`, `test / hooks` (pre-commit stage, all files), `test / docs` (`zensical build --strict`), `test / workflows` (MolCrafts/molcrafts-ci/actions/check-workflows), `test / result` | `test / python ({ubuntu,macos,windows}-latest, {3.12,3.14})` and the rest as in the fast tier | docs deploy: Cloudflare Pages, outside Actions |
| `release.yml` | — | — | `v*` tag: test.yml + `release / build` + `release / pypi` (only a `v*` tag on MolCrafts: `publish`); `workflow_dispatch` = dry run (no upload) |

`test / result` needs every other job and fails unless each succeeded or was
skipped; it is the one check the `ci` ruleset requires on `dev` and `master`.
The `protect-master` ruleset on `master` requires a pull request and blocks
force pushes and deletion. `.github/release.toml` declares the PyPI target.

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
