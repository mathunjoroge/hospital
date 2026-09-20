"""add source tracking to oncology_bookings

Revision ID: c8d3e6a1b405
Revises: d1a4c7e9f201
Create Date: 2026-09-20

Adds oncology_bookings.source so the oncology bookings board can distinguish
bookings made in the unit (ONCOLOGY) from bookings propagated from the
Records department's clinic catalog (RECORDS). Existing rows default to
ONCOLOGY.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "c8d3e6a1b405"
down_revision = "d1a4c7e9f201"
branch_labels = None
depends_on = None

_TABLE = "oncology_bookings"


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
                server_default="ONCOLOGY",
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
