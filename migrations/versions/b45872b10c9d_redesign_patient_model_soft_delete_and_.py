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
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()
    patient_cols = [c['name'] for c in inspector.get_columns('patients')] if 'patients' in tables else []

    with op.batch_alter_table('patients', schema=None) as batch_op:
        if 'is_active' not in patient_cols:
            batch_op.add_column(sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('1')))
        if 'deleted_at' not in patient_cols:
            batch_op.add_column(sa.Column('deleted_at', sa.DateTime(), nullable=True))
        if 'created_at' not in patient_cols:
            batch_op.add_column(sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')))
        if 'updated_at' not in patient_cols:
            batch_op.add_column(sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')))
        if 'created_by' not in patient_cols:
            batch_op.add_column(sa.Column('created_by', sa.Integer(), nullable=True))
            batch_op.create_foreign_key('fk_patients_created_by', 'users', ['created_by'], ['id'])
        if 'updated_by' not in patient_cols:
            batch_op.add_column(sa.Column('updated_by', sa.Integer(), nullable=True))
            batch_op.create_foreign_key('fk_patients_updated_by', 'users', ['updated_by'], ['id'])

    if 'patient_identifiers' not in tables:
        op.create_table('patient_identifiers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('patient_id', sa.String(length=20), nullable=False),
        sa.Column('identifier_type', sa.String(length=50), nullable=False),
        sa.Column('identifier_value', sa.String(length=100), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ),
        sa.PrimaryKeyConstraint('id')
        )
    if 'patient_merges' not in tables:
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
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()
    patient_cols = [c['name'] for c in inspector.get_columns('patients')] if 'patients' in tables else []

    with op.batch_alter_table('patients', schema=None) as batch_op:
        for col in ['updated_by', 'created_by', 'updated_at', 'created_at', 'deleted_at', 'is_active']:
            if col in patient_cols:
                batch_op.drop_column(col)

    if 'patient_identifiers' in tables:
        op.drop_table('patient_identifiers')
    if 'patient_merges' in tables:
        op.drop_table('patient_merges')
