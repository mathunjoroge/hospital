"""encrypt_national_id_column

Revision ID: c3f8a1d92e74
Revises: a681a9e2c0b9
Create Date: 2026-09-06

Migration:
  1. Widen national_id column from VARCHAR(50) → VARCHAR(500) to fit Fernet ciphertext.
  2. Remove the unique constraint on national_id (Fernet is non-deterministic; each
     encrypt() call produces a different ciphertext, so DB-level uniqueness is unenforceable).
  3. Encrypt all existing plaintext national_id values in place.
     - Idempotent: rows already starting with 'enc_v1:' are skipped.
     - Back-down: restores the column width but cannot reverse encryption of existing rows.
"""

import logging

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

# revision identifiers
revision = 'c3f8a1d92e74'
down_revision = 'a681a9e2c0b9'
branch_labels = None
depends_on = None


def _get_encrypt_fn():
    """Import encrypt_value lazily to avoid circular import issues at Alembic load time."""
    try:
        from departments.crypto import encrypt_value
        return encrypt_value
    except Exception as exc:
        logger.error(f"Could not import encrypt_value: {exc}")
        raise


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    unique_constraints = [uc['name'] for uc in inspector.get_unique_constraints('patients') if uc.get('name')]

    # ── Step 1: widen the column (VARCHAR 50 → 500) ──────────────────────────
    with op.batch_alter_table('patients') as batch_op:
        batch_op.alter_column(
            'national_id',
            existing_type=sa.String(50),
            type_=sa.String(500),
            existing_nullable=True,
        )
        if 'uq_patients_national_id' in unique_constraints:
            batch_op.drop_constraint('uq_patients_national_id', type_='unique')

    # ── Step 2: encrypt existing plaintext values (idempotent) ───────────────
    encrypt_value = _get_encrypt_fn()

    patients = bind.execute(
        sa.text("SELECT id, national_id FROM patients WHERE national_id IS NOT NULL")
    ).fetchall()

    encrypted_count = 0
    skipped_count = 0

    for row in patients:
        patient_id, national_id = row[0], row[1]

        # Idempotency guard — skip already-encrypted values
        if national_id.startswith('enc_v1:'):
            skipped_count += 1
            continue

        ciphertext = encrypt_value(national_id)
        bind.execute(
            sa.text("UPDATE patients SET national_id = :cipher WHERE id = :pid"),
            {'cipher': ciphertext, 'pid': patient_id},
        )
        encrypted_count += 1

    logger.info(
        f"national_id encryption: {encrypted_count} rows encrypted, "
        f"{skipped_count} rows already encrypted (skipped)."
    )


def downgrade():
    # Narrow the column back; existing data remains encrypted (irreversible).
    with op.batch_alter_table('patients') as batch_op:
        batch_op.alter_column(
            'national_id',
            existing_type=sa.String(500),
            type_=sa.String(50),
            existing_nullable=True,
        )
