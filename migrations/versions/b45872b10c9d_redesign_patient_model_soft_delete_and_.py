"""redesign_patient_model_soft_delete_and_identifiers

Revision ID: b45872b10c9d
Revises: 95c625106a63
Create Date: 2026-09-05 16:23:00.000000

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = 'b45872b10c9d'
down_revision = '95c625106a63'
branch_labels = None
depends_on = None


def upgrade():
    # Add soft-delete fields to patients and create new identifier/merge tables
    with op.batch_alter_table('patients', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_patients_name'), ['name'], unique=False)
        batch_op.create_index(batch_op.f('ix_patients_national_id'), ['national_id'], unique=True)
        batch_op.create_index(batch_op.f('ix_patients_patient_id'), ['patient_id'], unique=True)
    op.create_table('patient_identifiers',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('patient_id', sa.String(length=20), nullable=False),
    sa.Column('identifier_type', sa.String(length=50), nullable=False),
    sa.Column('identifier_value', sa.String(length=100), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('patient_merges',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('source_patient_id', sa.String(length=20), nullable=False),
    sa.Column('target_patient_id', sa.String(length=20), nullable=False),
    sa.Column('merged_by', sa.Integer(), nullable=False),
    sa.Column('merged_at', sa.DateTime(), nullable=False),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['merged_by'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade():
    # Reverse: drop new tables and remove soft-delete columns
    with op.batch_alter_table('patients', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_patients_patient_id'))
        batch_op.drop_index(batch_op.f('ix_patients_national_id'))
        batch_op.drop_index(batch_op.f('ix_patients_name'))
    op.drop_table('patient_identifiers')
    op.drop_table('patient_merges')
