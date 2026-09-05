# Contributing to HIMS

Thank you for contributing to the Health Information Management System (HIMS). To maintain high clinical safety, data privacy, and code quality, please adhere to the following development guidelines.

---

## 🛠️ Development Workflow

1. **Fork & Branching**:
   - Create feature/bugfix branches off `main` or `hardening/phase-0`.
   - Branch naming convention: `feature/short-description` or `fix/issue-description`.

2. **Coding Standards**:
   - Follow PEP 8 guidelines for Python code.
   - Use `ruff` for linting: `ruff check .`
   - Ensure all public functions and models include clear docstrings.
   - Do not commit plain text secrets, API keys, or raw patient records.

3. **Database Changes**:
   - Do not modify existing database tables directly without Flask-Migrate migration scripts.
   - Generate migration files: `flask db migrate -m "Description of change"`.
   - Test migrations both up and down before submitting pull requests.

4. **Testing Requirements**:
   - Every new endpoint or feature MUST include automated unit tests under `tests/`.
   - Maintain minimum 25% test coverage across the repository.
   - Run tests before pushing: `PYTHONPATH=. venv/bin/pytest -v`.

---

## 🔒 Security & Privacy Guidelines

- **Patient Identifiers**: Never print or log raw patient National IDs, phone numbers, or passwords.
- **Audit Logging**: Any action that accesses, creates, modifies, or deletes patient data MUST invoke `log_audit_event()`.
- **Role Control**: Ensure every route is decorated with `@login_required` and `@roles_required(...)`.

---

## 📬 Submitting Pull Requests

1. Rebase your branch onto `main`: `git rebase main`
2. Ensure CI linting and unit tests pass locally.
3. Open a Pull Request with a descriptive summary of changes and reference associated issue numbers.
