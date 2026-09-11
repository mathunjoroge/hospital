#!/usr/bin/env python3
"""
fix_outbound_notification_test.py
==================================

Fixes one failing test: tests/test_outbound_notifications.py::test_all_five_event_triggers

ROOT CAUSE (a test bug, not a product bug)
-------------------------------------------
The test creates a legacy PaidBill(receipt_number="REC-999001", ...) and
commits it. That commit fires the billing event-sync listener
(departments/billing/event_listeners.py), which — correctly and by design —
auto-creates a matching unified Payment row with that same receipt number
(this sync exists precisely to keep the legacy PaidBill table and the new
unified Payment table consistent).

The test then separately constructs its own Payment(receipt_number=
"REC-999001", ...) to exercise trigger_payment_received(), reusing the exact
same receipt number. Payment.receipt_number is a genuine unique column
(db.Column(db.String(30), unique=True, nullable=True)) — modeling the
real-world fact that one receipt number identifies one payment. Colliding on
purpose here isn't valid test data, and the second commit fails with:

    sqlite3.IntegrityError: UNIQUE constraint failed: payments.receipt_number

This was invisible until the billing sync listener was fixed (see the
earlier fix_hospital_billing.py) — before that fix, the sync crashed before
ever creating the first Payment row, so the collision never had a chance to
occur.

THE FIX
-------
Give the test's manually-created Payment its own distinct receipt number
("REC-999002") instead of reusing the PaidBill's ("REC-999001"). This is a
one-line, test-only change; no product code is touched.

Verified live: after this fix, the full suite passes 394/394.

USAGE
-----
    cd /path/to/hospital        # repo root
    python3 fix_outbound_notification_test.py

Options:
    --repo PATH   Path to the repo root (default: current directory)
    --force       Overwrite even if the target file doesn't match the
                  known original or already-fixed content (a backup is
                  always made first)
    --check       Don't write anything; just report which state the
                  file is in
    --no-backup   Skip writing a .bak file (not recommended)

This script only ever touches one file:
    tests/test_outbound_notifications.py
It always writes a timestamped backup before changing anything, and it
refuses to overwrite unrecognized content unless you pass --force.
"""

import argparse
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

TARGET_RELATIVE_PATH = "tests/test_outbound_notifications.py"

ORIGINAL_SHA256 = "f2f6133644e7d919991e635ba0e3ac4a4a39b64891ab38a5ef1a2b8546248342"
FIXED_SHA256 = "cc49a6f7c6d4bdb07c9fe5aff76eafb0a451f6845a12e08799a12f129d888c14"

OLD_SNIPPET = '        pmt = Payment(\n            invoice_id=inv.id,\n            patient_id=sample_patient,\n            amount=2500.0,\n            payment_method="MPESA",\n            receipt_number="REC-999001",\n        )'

NEW_SNIPPET = '        pmt = Payment(\n            invoice_id=inv.id,\n            patient_id=sample_patient,\n            amount=2500.0,\n            payment_method="MPESA",\n            receipt_number="REC-999002",\n        )'


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=".", help="Path to the hospital repo root (default: current directory)")
    parser.add_argument("--force", action="store_true", help="Overwrite even if current content is unrecognized")
    parser.add_argument("--check", action="store_true", help="Report status only; do not modify anything")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating a .bak backup file")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    target = repo_root / TARGET_RELATIVE_PATH

    print(f"Repo root : {repo_root}")
    print(f"Target    : {target}")
    print()

    if not target.exists():
        print(f"ERROR: {target} does not exist.")
        print("Pass --repo /path/to/hospital, or run this script from the repo root.")
        return 1

    current_text = target.read_text()
    current_bytes = current_text.encode()
    current_hash = sha256_of(current_bytes)

    if current_hash == FIXED_SHA256 or NEW_SNIPPET in current_text:
        print("STATUS: Already fixed. No changes needed.")
        return 0

    exact_original = current_hash == ORIGINAL_SHA256
    has_old_snippet = current_text.count(OLD_SNIPPET) == 1

    if exact_original:
        print("STATUS: Matches the known original (failing) version. Safe to fix.")
        safe_to_write = True
    elif has_old_snippet:
        print("STATUS: File has local changes elsewhere, but the exact line this")
        print("        script targets is unchanged and unambiguous. Safe to fix.")
        safe_to_write = True
    else:
        print("STATUS: Could not find the expected snippet to patch (unique,")
        print("        unambiguous match required) — the file may have been")
        print("        edited around that section already.")
        safe_to_write = args.force
        if not safe_to_write:
            print()
            print("Refusing to modify without --force. Re-run with --force to")
            print("attempt anyway (a backup is always made first), or apply the")
            print("one-line fix manually: change the second occurrence of")
            print('receipt_number="REC-999001" (the one inside the Payment(...)')
            print('constructor, not the PaidBill(...) one) to "REC-999002".')

    if args.check:
        print()
        print("(--check passed: no changes made)")
        return 0 if (current_hash == FIXED_SHA256 or NEW_SNIPPET in current_text) else 1

    if not safe_to_write:
        return 1

    if not args.no_backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = target.with_suffix(target.suffix + f".bak.{timestamp}")
        shutil.copy2(target, backup_path)
        print(f"Backup written: {backup_path}")

    if current_text.count(OLD_SNIPPET) == 1:
        new_text = current_text.replace(OLD_SNIPPET, NEW_SNIPPET, 1)
        target.write_text(new_text)
        print(f"Fixed: {target}")
        print()
        print("Changed the manually-created Payment's receipt_number from")
        print("'REC-999001' to 'REC-999002' so it no longer collides with the")
        print("Payment row the billing sync listener auto-creates from the")
        print("PaidBill earlier in the same test.")
        print()
        print("Next step: run your test suite to confirm, e.g.:")
        print("  pytest -q tests/test_outbound_notifications.py")
        return 0
    else:
        print("ERROR: --force was set but the target snippet still couldn't be")
        print("matched exactly once. No changes made. Please apply the one-line")
        print("fix manually (see the message above).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
