#!/usr/bin/env python3
"""
fix_lint_errors.py

Fixes the 22 ruff errors reported by:

    ruff check . --select E,F,W,I --ignore E501

on https://github.com/mathunjoroge/hospital

What it does
------------
1. Auto-formats/sorts imports and strips trailing whitespace on blank lines
   using `ruff check --fix` (covers all I001, F401, W293 findings).
2. Patches three real bugs that ruff flags but can't fix by itself:
     - departments/billing/routes.py   : `InvoiceLineItem` used but never imported
     - departments/billing/sync.py     : `sess` used but undefined (should be `db.session`)
     - departments/models/encounter.py : ALLOWED_STAGE_TRANSITIONS has 5 duplicate
                                          dict keys that silently shadow each other
3. Re-runs ruff to confirm the tree is clean.

Usage
-----
    python fix_lint_errors.py /path/to/hospital

If no path is given, it assumes the current directory is the repo root.
Requires `ruff` to be installed (`pip install ruff`).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Fix 1: departments/billing/routes.py — F821 Undefined name `InvoiceLineItem`
# ---------------------------------------------------------------------------
def fix_routes_missing_import(repo: Path) -> bool:
    path = repo / "departments" / "billing" / "routes.py"
    if not path.exists():
        print(f"  skip (not found): {path}")
        return False

    src = read(path)

    if "InvoiceLineItem" not in src:
        print("  skip: InvoiceLineItem not referenced")
        return False

    # Already imported? Nothing to do.
    if re.search(r"^\s*InvoiceLineItem\b", src, re.MULTILINE) and re.search(
        r"from departments\.models\.billing import \(([^)]*)\bInvoiceLineItem\b",
        src,
        re.DOTALL,
    ):
        print("  skip: InvoiceLineItem already imported")
        return False

    pattern = re.compile(
        r"from departments\.models\.billing import \(\n(?P<body>[^)]*)\)",
        re.DOTALL,
    )
    match = pattern.search(src)
    if not match:
        die(f"Could not locate 'from departments.models.billing import (...)' block in {path}")

    body = match.group("body")
    names = [line.strip().rstrip(",") for line in body.splitlines() if line.strip()]
    if "InvoiceLineItem" in names:
        print("  skip: InvoiceLineItem already imported")
        return False

    names.append("InvoiceLineItem")
    names = sorted(set(names))
    new_body = "".join(f"    {name},\n" for name in names)
    new_block = f"from departments.models.billing import (\n{new_body})"

    src = src[: match.start()] + new_block + src[match.end() :]
    write(path, src)
    print(f"  fixed: added InvoiceLineItem to import in {path.relative_to(repo)}")
    return True


# ---------------------------------------------------------------------------
# Fix 2: departments/billing/sync.py — F821 Undefined name `sess`
# ---------------------------------------------------------------------------
def fix_sync_undefined_sess(repo: Path) -> bool:
    path = repo / "departments" / "billing" / "sync.py"
    if not path.exists():
        print(f"  skip (not found): {path}")
        return False

    src = read(path)
    lines = src.splitlines(keepends=True)

    changed = False
    # Walk the file tracking whether we're inside a function that has its own
    # local `sess` (assigned via `sess = ...`). Only rewrite `sess.` usages
    # in functions where `sess` was never assigned locally.
    has_local_sess = False
    out = []

    def flush(block_lines, has_sess):
        nonlocal changed
        if has_sess:
            out.extend(block_lines)
            return
        for line in block_lines:
            if re.search(r"\bsess\.", line) and "def " not in line:
                new_line = re.sub(r"\bsess\.", "db.session.", line)
                if new_line != line:
                    changed = True
                out.append(new_line)
            else:
                out.append(line)

    block: list[str] = []
    for line in lines:
        if re.match(r"^def \w+\(", line):
            flush(block, has_local_sess)
            block = [line]
            has_local_sess = False
            continue
        if re.match(r"^\s*sess\s*=", line):
            has_local_sess = True
        block.append(line)
    flush(block, has_local_sess)

    if changed:
        write(path, "".join(out))
        print(f"  fixed: undefined `sess` -> `db.session` in {path.relative_to(repo)}")
    else:
        print("  skip: no undefined `sess` usage found")
    return changed


# ---------------------------------------------------------------------------
# Fix 3: departments/models/encounter.py — F601 repeated dict key literals
# ---------------------------------------------------------------------------
def fix_encounter_duplicate_keys(repo: Path) -> bool:
    path = repo / "departments" / "models" / "encounter.py"
    if not path.exists():
        print(f"  skip (not found): {path}")
        return False

    src = read(path)
    match = re.search(
        r"ALLOWED_STAGE_TRANSITIONS = \{\n(?P<body>.*?)\n\s*\}\n",
        src,
        re.DOTALL,
    )
    if not match:
        print("  skip: ALLOWED_STAGE_TRANSITIONS dict not found")
        return False

    body = match.group("body")

    # Split the dict body into one chunk per top-level "key": value entry,
    # keeping multi-line set literals (e.g. "IN_CONSULTATION": {...}) intact.
    entries = re.findall(
        r'(?:None|"[A-Z_]+")\s*:\s*(?:\{[^}]*\}|set\(\))\s*,',
        body,
        re.DOTALL,
    )

    seen = {}
    order = []
    for entry in entries:
        key_match = re.match(r"(None|\"[A-Z_]+\")\s*:", entry)
        key = key_match.group(1)
        if key in seen:
            order.remove(key)  # duplicate: drop earlier occurrence, keep latest
        seen[key] = entry
        order.append(key)

    if len(seen) == len(entries):
        print("  skip: no duplicate keys found")
        return False

    new_body_lines = []
    for key in order:
        entry = seen[key].strip()
        # Only the first line lost its indentation to strip(); continuation
        # lines of multi-line set literals already carry their original
        # (correct) indentation and must be kept verbatim.
        entry_lines = entry.splitlines()
        new_body_lines.append("        " + entry_lines[0])
        new_body_lines.extend(entry_lines[1:])
    new_body = "\n".join(new_body_lines)

    new_block = f"ALLOWED_STAGE_TRANSITIONS = {{\n{new_body}\n    }}\n"
    src = src[: match.start()] + new_block + src[match.end() :]
    write(path, src)
    removed = len(entries) - len(seen)
    print(f"  fixed: removed {removed} duplicate key(s) from ALLOWED_STAGE_TRANSITIONS in {path.relative_to(repo)}")
    return True


# ---------------------------------------------------------------------------
# Ruff auto-fix pass (handles I001, F401, W293) + verification
# ---------------------------------------------------------------------------
def run_ruff_fix(repo: Path) -> None:
    print("\nRunning `ruff check --fix` for import sorting / unused imports / whitespace...")
    result = subprocess.run(
        ["ruff", "check", ".", "--select", "E,F,W,I", "--ignore", "E501", "--fix"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def verify(repo: Path) -> bool:
    print("Verifying with `ruff check . --select E,F,W,I --ignore E501` ...")
    result = subprocess.run(
        ["ruff", "check", ".", "--select", "E,F,W,I", "--ignore", "E501"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def main() -> None:
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    if not repo.exists():
        die(f"Path does not exist: {repo}")
    if not (repo / "departments").exists():
        die(f"'{repo}' does not look like the hospital repo root (no departments/ dir)")

    try:
        subprocess.run(["ruff", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        die("ruff is not installed. Install it with: pip install ruff")

    print(f"Fixing lint errors in {repo}\n")

    print("[1/3] departments/billing/routes.py (F821 InvoiceLineItem)")
    fix_routes_missing_import(repo)

    print("\n[2/3] departments/billing/sync.py (F821 sess)")
    fix_sync_undefined_sess(repo)

    print("\n[3/3] departments/models/encounter.py (F601 duplicate keys)")
    fix_encounter_duplicate_keys(repo)

    run_ruff_fix(repo)

    ok = verify(repo)
    if ok:
        print("All clear: 0 ruff errors remaining.")
    else:
        print("Some errors remain — see output above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
