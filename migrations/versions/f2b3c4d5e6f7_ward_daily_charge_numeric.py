"""ward daily_charge Numeric safety migration

Revision ID: f2b3c4d5e6f7
Revises: f1a2b3c4d5e6
Create Date: 2026-09-15 12:00:00.000000

Changes:
  - wards.daily_charge: ensure Float → Numeric(10,2) schema declaration safety
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "f2b3c4d5e6f7"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("wards") as batch_op:
        batch_op.alter_column(
            "daily_charge",
            type_=sa.Numeric(10, 2),
            existing_type=sa.Numeric(10, 2),
            existing_nullable=False,
        )


def downgrade():
    with op.batch_alter_table("wards") as batch_op:
        batch_op.alter_column(
            "daily_charge",
            type_=sa.Numeric(10, 2),
            existing_type=sa.Numeric(10, 2),
            existing_nullable=False,
        )
