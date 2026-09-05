"""add_missing_columns_and_tables

Revision ID: a681a9e2c0b9
Revises: 49d0cf66035c
Create Date: 2026-09-05 17:42:46.570488

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a681a9e2c0b9'
down_revision = '49d0cf66035c'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('lab_results', schema=None) as batch_op:
        batch_op.add_column(sa.Column('status', sa.String(length=30), nullable=True))
        batch_op.add_column(sa.Column('panic_status', sa.String(length=30), nullable=True))
        batch_op.add_column(sa.Column('panic_message', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('verified_by', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('verified_at', sa.DateTime(), nullable=True))
        batch_op.create_foreign_key('fk_lab_results_verified_by_users', 'users', ['verified_by'], ['id'])

    with op.batch_alter_table('patients', schema=None) as batch_op:
        batch_op.add_column(sa.Column('insurance_provider', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('insurance_policy_number', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('occupation', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('employer_name', sa.String(length=100), nullable=True))


def downgrade():
    with op.batch_alter_table('patients', schema=None) as batch_op:
        batch_op.drop_column('employer_name')
        batch_op.drop_column('occupation')
        batch_op.drop_column('insurance_policy_number')
        batch_op.drop_column('insurance_provider')

    with op.batch_alter_table('lab_results', schema=None) as batch_op:
        batch_op.drop_constraint('fk_lab_results_verified_by_users', type_='foreignkey')
        batch_op.drop_column('verified_at')
        batch_op.drop_column('verified_by')
        batch_op.drop_column('panic_message')
        batch_op.drop_column('panic_status')
        batch_op.drop_column('status')
