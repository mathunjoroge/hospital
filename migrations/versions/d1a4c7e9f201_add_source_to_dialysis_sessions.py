"""add source tracking to dialysis_sessions

Revision ID: d1a4c7e9f201
Revises: 4c22bf975de7
Create Date: 2026-09-20

Adds dialysis_sessions.source so the renal console can distinguish sessions
logged in the unit (RENAL) from bookings propagated from the Records
department's clinic catalog (RECORDS). Existing rows default to RENAL.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "d1a4c7e9f201"
down_revision = "4c22bf975de7"
branch_labels = None
depends_on = None

_TABLE = "dialysis_sessions"


def _columns(inspector, table):
    return {c["name"] for c in inspector.get_columns(table)}


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _TABLE not in inspector.get_table_names():
        return
    existing = _columns(inspector, _TABLE)
    if "source" in existing:
        return
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "source",
                sa.String(length=20),
                nullable=False,
                server_default="RENAL",
            )
        )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _TABLE not in inspector.get_table_names():
        return
    existing = _columns(inspector, _TABLE)
    if "source" not in existing:
        return
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.drop_column("source")
