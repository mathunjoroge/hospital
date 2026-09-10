import subprocess
import sys
import ast
from pathlib import Path

def run_cmd(cmd):
    print(f"\n{'='*20} Executing: {cmd} {'='*20}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        sys.exit(result.returncode)
    print(f"✅ Command succeeded: {cmd}")

def add_import_safe(content, import_stmt):
    """AST-safe import insertion (after module docstring) to avoid Ruff E402."""
    if import_stmt.strip() in content:
        return content
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return import_stmt + '\n' + content
    lines = content.split('\n')
    insert_idx = 0
    if (tree.body and isinstance(tree.body[0], ast.Expr) and
            isinstance(getattr(tree.body[0], 'value', None), (ast.Constant, ast.Str))):
        insert_idx = getattr(tree.body[0], 'end_lineno', 1)
    else:
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith('#') or s == '':
                insert_idx = i + 1
            else:
                break
    lines.insert(insert_idx, import_stmt)
    return '\n'.join(lines)

def update_dispensing():
    print("\n📝 Wiring T3.7 enforcement into departments/pharmacy/dispensing.py...")
    file_path = Path('departments/pharmacy/dispensing.py')
    content = file_path.read_text(encoding='utf-8')

    # Idempotency guard
    if "# --- T3.7: Encounter Scoping Check ---" in content:
        print("✅ T3.7 enforcement already present. Skipping.")
        return True

    # Diagnostic: confirm `request` is available for request.referrer
    if 'request' not in content:
        print("⚠️ 'request' not found in dispensing.py — redirect may need adjustment.")
    else:
        print("  ℹ️ 'request' is imported/used. Good.")

    # AST-safe import of the helper
    content = add_import_safe(
        content,
        "from departments.shared.encounter_utils import is_encounter_open_for_dispensing",
    )

    # EXACT target block from grep (lines 393-397), 8-space indent
    target = """        prescribed_meds = (
            db.session.query(PrescribedMedicine)
            .filter_by(prescription_id=prescription_id)
            .all()
        )"""

    injection = """

        # --- T3.7: Encounter Scoping Check ---
        if prescribed_meds and not is_encounter_open_for_dispensing(prescribed_meds[0].encounter_id):
            flash('Cannot dispense: The associated encounter is closed or the patient has been discharged.', 'danger')
            return redirect(request.referrer or url_for('pharmacy.index'))
        # -------------------------------------"""

    if target not in content:
        print("❌ ERROR: exact target block not found. Aborting WITHOUT writing a broken file.")
        print("   Re-run the grep to confirm the current fetch site.")
        return False

    content = content.replace(target, target + injection, 1)
    file_path.write_text(content, encoding='utf-8')
    print("✅ T3.7 enforcement injected directly before the status=1 dispense update.")
    return True

if __name__ == "__main__":
    print("🚀 T3.7 (retry) — Wire pharmacy dispensing enforcement...")

    if not update_dispensing():
        sys.exit(1)

    run_cmd("python -m pytest -q")
    run_cmd("python -m ruff check . --fix")
    run_cmd("python -m ruff check .")

    run_cmd("git add departments/pharmacy/dispensing.py")
    run_cmd('git commit -m "feat: T3.7 - Enforce pharmacy dispensing block on closed/discharged encounters"')

    print("\n🎉 T3.7 enforcement now actually wired. Tests + ruff green, committed.")