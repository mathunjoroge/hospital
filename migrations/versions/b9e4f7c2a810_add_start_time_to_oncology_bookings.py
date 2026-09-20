"""add start_time to oncology_bookings

Revision ID: b9e4f7c2a810
Revises: c8d3e6a1b405
Create Date: 2026-09-20

Adds oncology_bookings.start_time (B5) so oncology bookings can carry an
optional slot time, mirroring the renal chair-time pattern. Existing rows
keep NULL (date-only bookings).
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "b9e4f7c2a810"
down_revision = "c8d3e6a1b405"
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
    if "start_time" in existing:
        return
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.add_column(sa.Column("start_time", sa.DateTime(), nullable=True))


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _TABLE not in inspector.get_table_names():
        return
    existing = _columns(inspector, _TABLE)
    if "start_time" not in existing:
        return
    with op.batch_alter_table(_TABLE, schema=None) as batch_op:
        batch_op.drop_column("start_time")
