"""add employee.user_id (link employee to login account)

Several self-service routes (leave_request, update_employee_profile,
employee_payslips, view_payslip, download_payslip) compared current_user.id
(a User primary key) directly against Employee.id (a separate table's
primary key) as if they were the same identifier. They aren't: Employee and
User are independent tables with independent auto-increment sequences, so
this only worked when a user's login account and employee record happened to
share a number by coincidence.

Live-verified impact before this fix: a real employee's own payslip list came
back empty, and viewing their own payslip directly returned a 302 denial,
because their User.id essentially never equals their Employee.id in any
system where the two are created independently over time.

This migration adds the missing link. It is nullable (not every employee has
a login — e.g. a contractor paid via payroll but with no system access) and
unique (a login account maps to at most one employee record).

No backfill is attempted here: there is no reliable signal in the existing
data to match a User row to an Employee row (username and employee name are
free text and not guaranteed to correspond). HR must set the link explicitly
per employee going forward, via the "Link to login account" field added to
the employee edit form in the accompanying code change.

Revision ID: c1d4f7a2b853
Revises: b3f8d2a6c941
Create Date: 2026-09-15
"""
import sqlalchemy as sa
from alembic import op

revision = "c1d4f7a2b853"
down_revision = "b3f8d2a6c941"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("employees", schema=None) as batch_op:
        batch_op.add_column(sa.Column("user_id", sa.Integer(), nullable=True))
        batch_op.create_index(
            batch_op.f("ix_employees_user_id"), ["user_id"], unique=True
        )
        batch_op.create_foreign_key(
            "fk_employees_user_id_users", "users", ["user_id"], ["id"]
        )


def downgrade():
    with op.batch_alter_table("employees", schema=None) as batch_op:
        batch_op.drop_constraint("fk_employees_user_id_users", type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_employees_user_id"))
        batch_op.drop_column("user_id")
