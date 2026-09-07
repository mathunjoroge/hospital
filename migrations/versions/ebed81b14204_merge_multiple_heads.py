"""merge_multiple_heads

Revision ID: ebed81b14204
Revises: 0f5def37a442, add_fk_indexes_20260907_205220
Create Date: 2026-09-07 21:07:11.158159

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ebed81b14204'
down_revision = ('0f5def37a442', 'add_fk_indexes_20260907_205220')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
