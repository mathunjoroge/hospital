"""Add snomed_codes and loinc_codes tables

Revision ID: 3b731701d177
Revises: 1a2b3c4d5e6f
Create Date: 2026-09-11 20:44:15.825519

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '3b731701d177'
down_revision = '1a2b3c4d5e6f'
branch_labels = None
depends_on = None


def upgrade():
    # Create snomed_codes table
    op.create_table(
        'snomed_codes',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('code', sa.String(50), unique=True, nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=False)
    )

    # Create loinc_codes table
    op.create_table(
        'loinc_codes',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('code', sa.String(50), unique=True, nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=False)
    )


def downgrade():
    op.drop_table('loinc_codes')
    op.drop_table('snomed_codes')


