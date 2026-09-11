"""add encounter_id to theatre_list for T3.2 surgical state machine

Revision ID: 98b602e8fcea
Revises: d20891842474
Create Date: 2026-09-10 16:00:00.000000

"""
import sqlalchemy as sa
from alembic import op

revision = '98b602e8fcea'
down_revision = 'd20891842474'
branch_labels = None
depends_on = None

def upgrade():
    with op.batch_alter_table('theatre_list', schema=None) as batch_op:
        batch_op.add_column(sa.Column('encounter_id', sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f('ix_theatre_list_encounter_id'), ['encounter_id'], unique=False)
        batch_op.create_foreign_key(None, 'encounters', ['encounter_id'], ['id'])

def downgrade():
    with op.batch_alter_table('theatre_list', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_theatre_list_encounter_id'))
        batch_op.drop_column('encounter_id')
