# Contributing to bami

This project uses a simple branch workflow so changes can be tested before they
become a released package.

## Branches

- `main` is the released version of the package. Do not commit directly to
  `main`.
- `develop` is the working branch for changes that have passed review but are
  not released yet.
- Feature branches are for one focused change at a time. Use names such as
  `codex/add-sdm-example` or `fix/checkpoint-paths`.

## Usual development flow

1. Start from the latest `develop` branch.
2. Create a feature branch for the change.
3. Make the smallest clear change that solves the problem.
4. Run the local checks.
5. Open a pull request into `develop`.
6. Merge into `develop` only after the checks pass.

Use these commands for local checks:

```bash
uv run pytest
uv run ruff check src tests
uv run black --check src tests
```

If Black reports formatting changes, run:

```bash
uv run black src tests
```

Then re-run the checks.

## Release flow

Releases happen from `main`. A release should collect the tested changes from
`develop`.

1. Make sure `develop` has the changes that should be released.
2. Create a release branch from `develop`.
3. Update `project.version` in `pyproject.toml`.
4. Run the local checks and build the package:

```bash
uv run pytest
uv run ruff check src tests
uv run black --check src tests
uv build
```

5. Open a pull request from the release branch into `main`.
6. Merge the pull request only after the checks pass.

When the release pull request is merged into `main`, GitHub Actions builds the
package, creates a tag named `v{version}`, and publishes a GitHub Release. For
example, version `0.1.1` becomes tag `v0.1.1`.

If a release with the same tag already exists, the release workflow fails. This
usually means `project.version` in `pyproject.toml` was not updated before the
merge.

## GitHub branch protection

In GitHub repository settings, protect both `main` and `develop`.

Recommended rules:

- Require a pull request before merging.
- Require the CI checks to pass.
- Block direct pushes to `main`.
- Use one merge style consistently, such as squash merge.

These rules prevent accidental releases and keep the package history easier to
understand.
