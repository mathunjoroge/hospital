"""drop unused custom_rule table

departments/models/hr.py::CustomRule had zero readers and zero writers
anywhere in the codebase — no route ever queried it, and nothing ever
inserted into it (confirmed by a full-codebase search before removal). It
appears to have been an early, more general design for per-employee or
per-job-group payroll rules (it has both an employee_id and a job_group
column, plus a type field distinguishing deduction/allowance) that was
superseded by the simpler Allowance (job-group-keyed) and Deduction
(platform-wide) models actually used by generate_payroll().

Safe to drop outright: no foreign key points at custom_rule, and with no
writers there is no data to migrate anywhere (unlike the consents ->
patient_consents consolidation in a7c3e91d4b28, where a real writer existed
and rows had to be translated across).

Revision ID: d29e6b4f1a73
Revises: c1d4f7a2b853
Create Date: 2026-09-16
"""
import sqlalchemy as sa
from alembic import op

revision = "d29e6b4f1a73"
down_revision = "c1d4f7a2b853"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "custom_rule" not in inspector.get_table_names():
        return

    existing_indexes = {ix["name"] for ix in inspector.get_indexes("custom_rule")}
    with op.batch_alter_table("custom_rule", schema=None) as batch_op:
        if "ix_custom_rule_employee_id" in existing_indexes:
            batch_op.drop_index("ix_custom_rule_employee_id")
    op.drop_table("custom_rule")


def downgrade():
    op.create_table(
        "custom_rule",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("employee_id", sa.Integer(), nullable=True),
        sa.Column("job_group", sa.String(length=50), nullable=True),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("is_percentage", sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(["employee_id"], ["employees.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("custom_rule", schema=None) as batch_op:
        batch_op.create_index(
            "ix_custom_rule_employee_id", ["employee_id"], unique=False
        )
