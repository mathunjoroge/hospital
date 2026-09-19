# Contributing to HIMS

Thank you for helping improve the Health Information Management System (HIMS). Because this software handles sensitive health data and supports clinical workflows, we hold contributions to a high bar for **patient safety, data privacy and code quality**. This guide explains how to contribute and what to expect in review.

By contributing, you agree that your contributions are released under the project's [MIT License](LICENSE). Please be respectful and constructive in issues, pull requests and discussions.

## Contents

- [Getting started](#getting-started)
- [Development workflow](#development-workflow)
- [Checks to run before you push](#checks-to-run-before-you-push)
- [Coding standards](#coding-standards)
- [Database changes](#database-changes)
- [Testing](#testing)
- [Security and privacy](#security-and-privacy)
- [Clinical safety](#clinical-safety)
- [Dependencies](#dependencies)
- [AI-assisted contributions](#ai-assisted-contributions)
- [Submitting a pull request](#submitting-a-pull-request)

---

## Getting started

1. **Find or open an issue.** Look for issues labelled `good first issue` or `help wanted`. For anything larger than a small fix, please open an issue and discuss your approach before writing code.
2. **Set up your environment.** Follow the [Quickstart in the README](README.md#quickstart) to run the app locally. Copy `.env.example` to `.env` for local configuration:
   ```bash
   cp .env.example .env
   ```
   Never commit `.env`, real secrets or credentials.
3. **Report security problems privately.** Do not open public issues for vulnerabilities. See [`SECURITY.md`](SECURITY.md).

---

## Development workflow

1. Fork the repository and clone your fork. Add the main repository as `upstream`:
   ```bash
   git remote add upstream https://github.com/mathunjoroge/hospital.git
   ```
2. Create a branch from the latest `main`:
   ```bash
   git fetch upstream
   git checkout -b feature/short-description upstream/main
   ```
   Use `feature/short-description` for new work and `fix/issue-description` for bug fixes.
3. Make focused commits with clear messages. Keep each pull request to one logical change.
4. Before opening a PR, rebase onto the latest upstream `main`:
   ```bash
   git fetch upstream
   git rebase upstream/main
   git push --force-with-lease origin your-branch
   ```

---

## Checks to run before you push

Our CI runs the following on every push and pull request. Run the same commands locally so you don't wait for a failed build:

```bash
# Lint
ruff check . --select E,F,W,I --ignore E501

# Static security scan (medium severity and above)
bandit -r . -x ./venv,./tests,./migrations --severity-level medium -q

# Dependency vulnerability audit
pip-audit

# Tests with coverage
PYTHONPATH=. pytest --cov=departments --cov=app --cov-report=term-missing --cov-fail-under=45 -v
```

CI also enforces:

- **Zero test failures or errors.**
- **A minimum total test count.** The suite must not shrink. If your change legitimately removes tests, explain why in the PR so a maintainer can update the baseline.
- **Coverage of at least 45%** across `departments` and `app`. This is a floor, not a target: all new or changed code should be covered by tests.

A pull request cannot be merged until CI passes.

---

## Coding standards

- Follow PEP 8. Import ordering and the lint rules above are enforced by `ruff`.
- Give public functions, classes and models clear docstrings.
- Keep clinical, billing and privacy logic explicit and readable. Prefer clarity over cleverness.
- Never commit plain-text secrets, API keys or real patient records.

---

## Database changes

- Never alter tables directly. Every schema change needs a Flask-Migrate migration:
  ```bash
  flask db migrate -m "Describe the change"
  ```
- **Review the generated migration by hand.** Autogeneration can miss renames, data migrations and constraints.
- Test both directions before submitting:
  ```bash
  flask db upgrade
  flask db downgrade
  flask db upgrade
  ```
- Never edit a migration that has already been merged. Add a new one instead.
- CI runs against SQLite, while production targets PostgreSQL. If your migration uses database-specific behaviour, test it on PostgreSQL and say so in the PR.

---

## Testing

- Every new endpoint or feature **must** include automated tests under `tests/`.
- Bug fixes should include a test that fails without the fix.
- Use **synthetic data only** in tests and fixtures. Never use real patient information, even anonymised.

---

## Security and privacy

- **Patient identifiers:** never print or log raw National IDs, phone numbers, passwords or other identifying data.
- **Audit logging:** any action that reads, creates, modifies or deletes patient data must call `log_audit_event()`. Follow the existing call sites for the expected arguments.
- **Access control:** protect every route with `@login_required` and `@roles_required(...)` using the narrowest role that works. Routes that are intentionally public (for example `/healthz`) need an explicit justification in the PR.
- **Encryption:** sensitive identity fields are encrypted at rest. Don't bypass or weaken field-level encryption.
- **No real data anywhere:** issues, PR descriptions, screenshots, logs and test files must not contain real patient data or real credentials. If you accidentally post any, tell a maintainer immediately.

---

## Clinical safety

Changes that affect clinical behaviour carry extra risk. This includes triage scoring, drug interaction or dosing checks, lab reference ranges and critical values, ICD-10 coding, and prescribing validation.

For these changes:

- Cite the clinical source or guideline your change is based on, in the PR description.
- Add tests covering normal, boundary and unsafe cases.
- Expect review from a maintainer or clinical reviewer before merge, which may take longer than usual.

---

## Dependencies

- Justify any new dependency in the PR, and prefer well-maintained packages with compatible licenses.
- New dependencies must pass `pip-audit`. CI ignores a small, documented list of advisories; please don't add to that list without maintainer agreement.

---

## AI-assisted contributions

AI coding tools are welcome. You remain fully responsible for what you submit: understand the code, run the checks above, and test it yourself. Don't paste real patient data or secrets into AI tools, and don't submit generated changes to clinical logic without verifying them against a clinical source.

---

## Submitting a pull request

Open a pull request against `main` with:

- A clear summary of what changed and why.
- References to related issues (for example `Closes #123`).
- Notes on how you tested the change.

Before requesting review, confirm:

- [ ] Branch is rebased on the latest upstream `main`
- [ ] Lint, bandit, pip-audit and tests pass locally, and CI is green
- [ ] New or changed behaviour has tests, using synthetic data only
- [ ] Database changes include a reviewed migration, tested up and down
- [ ] Routes are protected with `@login_required` and `@roles_required(...)`, or the public access is justified
- [ ] Patient-data access and changes call `log_audit_event()`
- [ ] No secrets, real patient data or personal identifiers in code, logs, screenshots or tests
- [ ] Clinical changes cite a source