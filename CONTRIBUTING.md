# Contributing to BioMedStatX

First off — thank you for your interest in contributing to BioMedStatX. Contributions make open-source projects better for everyone and are always welcome.

This document explains how to contribute, the workflow we expect, coding style, testing expectations, how to add new statistical functions, and the review process. Please also note that the project is distributed under the MIT License — see the LICENSE file.

---

## Table of contents

- Getting started
- Fork-and-branch workflow
- Branch naming
- Commit messages
- Pull request (PR) structure and expectations
- Code style and quality tools
- Tests and testing expectations
- Guidelines for adding new statistical functions
- Review and merging process
- Reporting security issues

---

## Getting started

1. Fork the repository to your GitHub account.
2. Clone your fork:
   git clone https://github.com/<your-username>/BioMedStatX.git
3. Add the main repo as upstream:
   git remote add upstream https://github.com/philippkrumm/BioMedStatX.git
4. Create a topic branch for your work (see Branch naming below).

Always work in your fork and topic branches. Do not commit directly to the `main` (or `master`) branch.

---

## Fork-and-branch workflow

- Fork the repository.
- Create a new branch in your fork for each logical change:
  - git checkout -b feature/describe-what-you-are-changing
- Make changes, run tests and linters locally.
- Commit changes with clear messages and push to your fork.
- Open a Pull Request from your fork/branch to the main repository.

---

## Branch naming conventions

Use descriptive branch names using one of the following prefixes:

- feature/<short-description> — new features or non-breaking improvements
- fix/<short-description> — bug fixes
- docs/<short-description> — documentation changes
- refactor/<short-description> — internal code changes / refactors
- test/<short-description> — tests only
- ci/<short-description> — CI or automation updates

Example: feature/add-mannwhitney-helper

---

## Commit message guidelines

- Use present tense: "Add", "Fix", "Update".
- Keep the subject line short (~50 characters).
- Provide a blank line then a more detailed body if needed.
- Reference related issue numbers: "Fixes #123".

Example:
Add utility to compute pairwise effect sizes

This adds a new helper function `effect_size` used by the post-hoc modules.
It includes unit tests and documentation updates.

---

## Pull request (PR) structure and expectations

When you open a PR, please include:

- A concise title describing the change.
- A description summarizing:
  - What you changed and why.
  - Any design decisions that matter.
  - A checklist describing what you tested.
- Link to any related issues (e.g., "Fixes #123").
- Where appropriate, include screenshots, examples, or small reproducible snippets.
- Ensure the PR modifies only files relevant to the change.

Recommended PR template (informal):

- Title: short descriptive title
- Body:
  - Summary of changes
  - Related issues
  - Checklist:
    - [ ] My code follows the project’s style
    - [ ] I added tests where applicable
    - [ ] I updated documentation where applicable
    - [ ] I ran the test suite: pytest -q

A maintainer may request changes; please respond to review comments promptly.

---

## Code style and quality

This project follows standard Python open-source practices:

- Follow PEP 8 for code formatting.
- Use black for automated formatting (recommended configuration: line-length 88 or 100 as used in the project if specified).
- Use isort for import sorting.
- Add or update type hints where appropriate.
- Prefer clear, well-documented functions with concise docstrings (NumPy or Google style is acceptable).
- Keep functions small and focused.

Suggested tooling (developers should run locally):

- Install development dependencies (project may maintain a `requirements-dev.txt` or `pyproject.toml`).
- Run formatters: black .
- Run linters: flake8 .
- Run import sorter: isort .

If the repository adds a pre-commit configuration, install pre-commit hooks:
- pip install pre-commit
- pre-commit install

---

## Tests and testing expectations

- Tests should be added using pytest and placed in the `tests/` directory (matching the module structure when possible).
- Each new statistical function or algorithm must include unit tests that:
  - Cover expected behavior for normal inputs.
  - Exercise edge cases (empty inputs, extreme values, ties, NaNs if supported).
  - Check numerical stability where applicable, e.g., using tolerances in assertions.
- Use deterministic inputs in tests (set random seeds).
- If the change affects numerical results, include tests that assert results within tight tolerances or compare to a reliable reference implementation (e.g., SciPy).
- Run the test suite locally: pytest -q

Target: keep tests fast and deterministic. Long-running experiments or benchmarks should be in `benchmarks/` or `examples/` and not run in CI by default.

---

## Guidelines for adding new statistical functions

When adding statistical procedures (e.g., new tests, estimators, post-hoc analyses), follow these steps:

1. Discuss the design in an Issue or in a PR describing the proposed API and algorithms.
2. Implement the function/module in a logically named module and include documentation strings and examples.
3. Add unit tests that cover correctness, edge cases, and numerical stability.
4. If applicable, add a small example in the `docs/` folder showing typical usage.
5. Reference the original source or paper in the docstring (citation or DOI).
6. Ensure inputs and outputs follow the existing project conventions (data formats, return types).
7. Add validation for inputs and clear error messages on misuse.
8. For computationally heavy routines, consider providing both a reference implementation (easy-to-follow) and an optimized implementation later; include tests for both.

---

## Documentation

- Update `docs/` when introducing user-facing behavior.
- Add examples to the HowTo guide where appropriate.

---

## Review and merging process

- PRs will be reviewed by maintainers or core contributors.
- A PR requires at least one approving review and passing CI checks to be merged.
- Maintain semantic changes separate from cosmetic/documentation changes where possible.
- For large changes, maintainers may request smaller PRs for incremental review.

---

## Reporting security issues

Do not open public issues for security vulnerabilities. Contact the maintainers privately (use the repository owner email listed in the README) for disclosure. We will coordinate a fix and disclosure timeline.

---

## LICENSE

This project is licensed under the MIT License. See the LICENSE file in the repository root.
