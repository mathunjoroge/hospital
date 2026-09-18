"""hr: add contract/NOK fields to employees, add leave_balances, performance_reviews,
training_records, disciplinary_records tables.

Revision ID: a1b2c3d4e5f6
Revises: fefcbf381ba9
Create Date: 2026-09-18

Changes
-------
Employee table
  - contract_type       VARCHAR(30)   default 'permanent'
  - probation_end_date  DATE
  - national_id         VARCHAR(30)
  - kra_pin             VARCHAR(20)
  - nok_name            VARCHAR(100)
  - nok_phone           VARCHAR(20)
  - nok_relationship    VARCHAR(50)
  - Leave.reason        TEXT          (nullable)
  - Leave.approved_by   FK → users.id
  - Leave.approved_at   DATETIME

New tables
  - leave_balances
  - performance_reviews
  - training_records
  - disciplinary_records
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "b2c3d4e5f6a7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    # ── Employee: new columns ────────────────────────────────────────────
    emp_cols = [c["name"] for c in inspector.get_columns("employees")] if "employees" in tables else []
    with op.batch_alter_table("employees") as batch_op:
        if "contract_type" not in emp_cols:
            batch_op.add_column(sa.Column("contract_type", sa.String(30), nullable=True, server_default="permanent"))
        if "probation_end_date" not in emp_cols:
            batch_op.add_column(sa.Column("probation_end_date", sa.Date(), nullable=True))
        if "national_id" not in emp_cols:
            batch_op.add_column(sa.Column("national_id", sa.String(30), nullable=True))
        if "kra_pin" not in emp_cols:
            batch_op.add_column(sa.Column("kra_pin", sa.String(20), nullable=True))
        if "nok_name" not in emp_cols:
            batch_op.add_column(sa.Column("nok_name", sa.String(100), nullable=True))
        if "nok_phone" not in emp_cols:
            batch_op.add_column(sa.Column("nok_phone", sa.String(20), nullable=True))
        if "nok_relationship" not in emp_cols:
            batch_op.add_column(sa.Column("nok_relationship", sa.String(50), nullable=True))

    # ── Leave: additional columns ────────────────────────────────────────
    leave_cols = [c["name"] for c in inspector.get_columns("leaves")] if "leaves" in tables else []
    with op.batch_alter_table("leaves") as batch_op:
        if "reason" not in leave_cols:
            batch_op.add_column(sa.Column("reason", sa.Text(), nullable=True))
        if "approved_by" not in leave_cols:
            batch_op.add_column(sa.Column("approved_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True))
        if "approved_at" not in leave_cols:
            batch_op.add_column(sa.Column("approved_at", sa.DateTime(), nullable=True))

    # ── leave_balances ───────────────────────────────────────────────────
    if "leave_balances" not in tables:
        op.create_table(
            "leave_balances",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=False),
            sa.Column("year", sa.Integer(), nullable=False),
            sa.Column("leave_type", sa.String(50), nullable=False),
            sa.Column("entitled_days", sa.Integer(), nullable=False, server_default="21"),
            sa.Column("used_days", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("carried_over", sa.Integer(), nullable=False, server_default="0"),
            sa.UniqueConstraint("employee_id", "year", "leave_type", name="uq_leave_balance"),
        )

    # ── performance_reviews ──────────────────────────────────────────────
    if "performance_reviews" not in tables:
        op.create_table(
            "performance_reviews",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=False),
            sa.Column("reviewer_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("review_period", sa.String(50), nullable=False),
            sa.Column("review_type", sa.String(30), nullable=False, server_default="annual"),
            sa.Column("score", sa.Integer(), nullable=False),
            sa.Column("strengths", sa.Text(), nullable=True),
            sa.Column("areas_for_improvement", sa.Text(), nullable=True),
            sa.Column("goals_next_period", sa.Text(), nullable=True),
            sa.Column("comments", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )

    # ── training_records ─────────────────────────────────────────────────
    if "training_records" not in tables:
        op.create_table(
            "training_records",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=False),
            sa.Column("title", sa.String(200), nullable=False),
            sa.Column("provider", sa.String(200), nullable=True),
            sa.Column("training_type", sa.String(50), nullable=False, server_default="cpd"),
            sa.Column("date_completed", sa.Date(), nullable=False),
            sa.Column("expiry_date", sa.Date(), nullable=True),
            sa.Column("cpd_points", sa.Integer(), nullable=True),
            sa.Column("certificate_number", sa.String(100), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("recorded_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )

    # ── disciplinary_records ─────────────────────────────────────────────
    if "disciplinary_records" not in tables:
        op.create_table(
            "disciplinary_records",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=False),
            sa.Column("incident_date", sa.Date(), nullable=False),
            sa.Column("incident_type", sa.String(50), nullable=False),
            sa.Column("description", sa.Text(), nullable=False),
            sa.Column("action_taken", sa.Text(), nullable=False),
            sa.Column("outcome", sa.String(50), nullable=True),
            sa.Column("reviewed_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        )


def downgrade():
    op.drop_table("disciplinary_records")
    op.drop_table("training_records")
    op.drop_table("performance_reviews")
    op.drop_table("leave_balances")

    with op.batch_alter_table("leaves") as batch_op:
        batch_op.drop_column("approved_at")
        batch_op.drop_column("approved_by")
        batch_op.drop_column("reason")

    with op.batch_alter_table("employees") as batch_op:
        batch_op.drop_column("nok_relationship")
        batch_op.drop_column("nok_phone")
        batch_op.drop_column("nok_name")
        batch_op.drop_column("kra_pin")
        batch_op.drop_column("national_id")
        batch_op.drop_column("probation_end_date")
        batch_op.drop_column("contract_type")
