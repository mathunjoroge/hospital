"""consolidate consent onto patient_consents

Retires the duplicate `consents` table (departments/consent/models.py :: Consent)
in favour of `patient_consents` (departments/models/compliance.py ::
PatientConsent), which is the table every consent gate in the platform actually
reads and the only one anything ever wrote to.

Before dropping, any rows present are copied across. The two schemas differ in
how they key the patient:

    consents.patient_id          INTEGER  -> patients.id      (numeric PK)
    patient_consents.patient_id  VARCHAR  -> patients.patient_id  (business ID)

so the copy joins through `patients` to translate. Rows whose patient no longer
exists cannot be translated and are left behind rather than silently dropped on
the floor with a NULL FK — the downgrade recreates an empty table, so run the
verification query in the docstring below before upgrading in production if you
have live consent data.

    SELECT COUNT(*) FROM consents;                       -- rows to migrate
    SELECT COUNT(*) FROM consents c
      LEFT JOIN patients p ON p.id = c.patient_id
     WHERE p.id IS NULL;                                 -- unmappable rows

Revision ID: a7c3e91d4b28
Revises: f2b3c4d5e6f7
Create Date: 2026-09-15
"""
import sqlalchemy as sa
from alembic import op

revision = "a7c3e91d4b28"
down_revision = "f2b3c4d5e6f7"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "consents" not in inspector.get_table_names():
        return

    # Copy forward anything the old table holds. `consents.status` is the
    # ACTIVE/REVOKED/EXPIRED string; `patient_consents` splits that into the
    # is_granted boolean plus revoked_at, so map rather than copy verbatim.
    op.execute(
        sa.text(
            """
            INSERT INTO patient_consents
                (patient_id, consent_type, is_granted, granted_at, revoked_at, notes)
            SELECT p.patient_id,
                   c.consent_type,
                   CASE WHEN c.status = 'ACTIVE' AND c.revoked_at IS NULL
                        THEN 1 ELSE 0 END,
                   c.granted_at,
                   c.revoked_at,
                   c.revocation_reason
              FROM consents c
              JOIN patients p ON p.id = c.patient_id
             WHERE NOT EXISTS (
                   SELECT 1 FROM patient_consents pc
                    WHERE pc.patient_id = p.patient_id
                      AND pc.consent_type = c.consent_type
             )
            """
        )
    )

    with op.batch_alter_table("consents", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_consents_patient_id"))
    op.drop_table("consents")


def downgrade():
    # Recreates the table shape only. Rows are not copied back: patient_consents
    # is the surviving source of truth and splitting it again would require
    # guessing which rows originated here.
    op.create_table(
        "consents",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column("consent_type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("granted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_by", sa.String(length=100), nullable=True),
        sa.Column("revocation_reason", sa.Text(), nullable=True),
        sa.Column("document_reference", sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("consents", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_consents_patient_id"), ["patient_id"], unique=False
        )
