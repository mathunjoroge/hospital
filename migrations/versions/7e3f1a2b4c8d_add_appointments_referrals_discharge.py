"""Add appointments, referrals, and discharge_summaries tables

Revision ID: 7e3f1a2b4c8d
Revises: a84b0e726071
Create Date: 2026-09-07 23:46:00

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '7e3f1a2b4c8d'
down_revision = 'a84b0e726071'
branch_labels = None
depends_on = None


def upgrade():
    # Guard: skip tables that already exist (safe to run multiple times)
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = inspector.get_table_names()

    if 'appointments' not in existing:
        op.create_table(
            'appointments',
            sa.Column('id', sa.String(length=36), nullable=False),
            sa.Column('patient_id', sa.Integer(), nullable=False),
            sa.Column('provider_id', sa.Integer(), nullable=False),
            sa.Column('scheduled_start', sa.DateTime(timezone=True), nullable=False),
            sa.Column('scheduled_end', sa.DateTime(timezone=True), nullable=False),
            sa.Column('status', sa.String(length=20), nullable=False, server_default='SCHEDULED'),
            sa.Column('appointment_type', sa.String(length=50), nullable=False),
            sa.Column('reason_for_visit', sa.Text(), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('appointments', schema=None) as batch_op:
            batch_op.create_index('ix_appointments_patient_id', ['patient_id'], unique=False)
            batch_op.create_index('ix_appointments_provider_id', ['provider_id'], unique=False)
            batch_op.create_index('ix_appointments_scheduled_start', ['scheduled_start'], unique=False)

    if 'referrals' not in existing:
        op.create_table(
            'referrals',
            sa.Column('id', sa.String(length=36), nullable=False),
            sa.Column('patient_id', sa.Integer(), nullable=False),
            sa.Column('referring_facility', sa.String(length=150), nullable=False),
            sa.Column('receiving_facility', sa.String(length=150), nullable=False),
            sa.Column('clinical_summary', sa.Text(), nullable=False),
            sa.Column('reason_for_referral', sa.String(length=255), nullable=False),
            sa.Column('status', sa.String(length=20), nullable=False, server_default='PENDING'),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('referrals', schema=None) as batch_op:
            batch_op.create_index('ix_referrals_patient_id', ['patient_id'], unique=False)

    if 'discharge_summaries' not in existing:
        op.create_table(
            'discharge_summaries',
            sa.Column('id', sa.String(length=36), nullable=False),
            sa.Column('patient_id', sa.Integer(), nullable=False),
            sa.Column('appointment_id', sa.String(length=36), nullable=True),
            sa.Column('admission_date', sa.DateTime(timezone=True), nullable=False),
            sa.Column('discharge_date', sa.DateTime(timezone=True), nullable=False),
            sa.Column('primary_diagnosis', sa.String(length=255), nullable=False),
            sa.Column('secondary_diagnoses', sa.Text(), nullable=True),
            sa.Column('discharge_medications', sa.Text(), nullable=True),
            sa.Column('follow_up_instructions', sa.Text(), nullable=True),
            sa.Column('referred_to', sa.String(length=150), nullable=True),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('discharge_summaries', schema=None) as batch_op:
            batch_op.create_index('ix_discharge_summaries_patient_id', ['patient_id'], unique=False)


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = inspector.get_table_names()

    if 'discharge_summaries' in existing:
        with op.batch_alter_table('discharge_summaries', schema=None) as batch_op:
            batch_op.drop_index('ix_discharge_summaries_patient_id')
        op.drop_table('discharge_summaries')

    if 'referrals' in existing:
        with op.batch_alter_table('referrals', schema=None) as batch_op:
            batch_op.drop_index('ix_referrals_patient_id')
        op.drop_table('referrals')

    if 'appointments' in existing:
        with op.batch_alter_table('appointments', schema=None) as batch_op:
            batch_op.drop_index('ix_appointments_scheduled_start')
            batch_op.drop_index('ix_appointments_provider_id')
            batch_op.drop_index('ix_appointments_patient_id')
        op.drop_table('appointments')
