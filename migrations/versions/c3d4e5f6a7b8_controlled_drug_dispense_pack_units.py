"""add pack_units_qty column to controlled_drug_dispenses table (Finding D)

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-09-18

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "c3d4e5f6a7b8"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "controlled_drug_dispenses" in tables:
        cols = [c["name"] for c in inspector.get_columns("controlled_drug_dispenses")]
        if "pack_units_qty" not in cols:
            op.add_column(
                "controlled_drug_dispenses",
                sa.Column("pack_units_qty", sa.Integer(), nullable=True),
            )
        if "batch_id" not in cols:
            op.add_column(
                "controlled_drug_dispenses",
                sa.Column("batch_id", sa.Integer(), sa.ForeignKey("batches.id"), nullable=True),
            )


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if "controlled_drug_dispenses" in tables:
        cols = [c["name"] for c in inspector.get_columns("controlled_drug_dispenses")]
        if "batch_id" in cols:
            op.drop_column("controlled_drug_dispenses", "batch_id")
        if "pack_units_qty" in cols:
            op.drop_column("controlled_drug_dispenses", "pack_units_qty")

