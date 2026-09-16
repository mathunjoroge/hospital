"""add employee.basic_salary

generate_payroll() has always computed gross_pay as
`employee.basic_salary + total_allowances`, but Employee never had a
basic_salary column — every call to generate_payroll() raised AttributeError
before writing a single Payroll row. The route also has no UI entry point
today (see the accompanying HR fix), so this has apparently never run outside
of direct URL access.

Nullable rather than NOT NULL: a newly hired employee without a salary set
yet is a legitimate state, not bad data. generate_payroll() now skips
employees with no basic_salary and flashes which ones were skipped, instead
of crashing the whole payroll run.

Revision ID: b3f8d2a6c941
Revises: a7c3e91d4b28
Create Date: 2026-09-15
"""
import sqlalchemy as sa
from alembic import op

revision = "b3f8d2a6c941"
down_revision = "a7c3e91d4b28"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("employees", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("basic_salary", sa.Numeric(precision=12, scale=2), nullable=True)
        )


def downgrade():
    with op.batch_alter_table("employees", schema=None) as batch_op:
        batch_op.drop_column("basic_salary")
