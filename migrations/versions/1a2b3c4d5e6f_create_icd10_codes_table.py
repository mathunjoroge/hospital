"""create icd10_codes table

Revision ID: 1a2b3c4d5e6f
Revises: e47ef5938c09
Create Date: 2026-09-11 21:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1a2b3c4d5e6f'
down_revision = 'e47ef5938c09'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'icd10_codes',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('code', sa.String(20), unique=True, nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('chapter', sa.String(100), nullable=True),
        sa.Column('block', sa.String(100), nullable=True)
    )


def downgrade():
    op.drop_table('icd10_codes')


