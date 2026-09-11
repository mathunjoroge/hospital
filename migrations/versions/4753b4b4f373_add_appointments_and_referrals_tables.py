"""Add appointments, referrals, and discharge_summaries tables

These models (departments/appointments/models.py and
departments/referrals/models.py) existed in the codebase and were being
imported by their blueprints, but had never been added to a migration,
so the tables never existed in any database. This backs the "Live Queue"
and "Referrals" screens now that their blueprints are registered in app.py.

Note: patient_id / provider_id on these tables are plain indexed integers
with no FK constraint, matching the existing style of the other
"engine" modules (clinical_safety, telemedicine). This is a different
convention from the string patient_id FK used by records/billing/consent
(Patient.patient_id, e.g. "P0001") — see DECISIONS_PENDING.md.

Revision ID: 4753b4b4f373
Revises: a84b0e726071
Create Date: 2026-09-07

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '4753b4b4f373'
down_revision = 'a84b0e726071'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'appointments',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('patient_id', sa.Integer(), nullable=False),
        sa.Column('provider_id', sa.Integer(), nullable=False),
        sa.Column('scheduled_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('scheduled_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('appointment_type', sa.String(length=50), nullable=False),
        sa.Column('reason_for_visit', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('appointments', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_appointments_patient_id'), ['patient_id'], unique=False
        )
        batch_op.create_index(
            batch_op.f('ix_appointments_provider_id'), ['provider_id'], unique=False
        )
        batch_op.create_index(
            batch_op.f('ix_appointments_scheduled_start'),
            ['scheduled_start'],
            unique=False,
        )

    op.create_table(
        'referrals',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('patient_id', sa.Integer(), nullable=False),
        sa.Column('referring_facility', sa.String(length=150), nullable=False),
        sa.Column('receiving_facility', sa.String(length=150), nullable=False),
        sa.Column('clinical_summary', sa.Text(), nullable=False),
        sa.Column('reason_for_referral', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('referrals', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_referrals_patient_id'), ['patient_id'], unique=False
        )

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
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('discharge_summaries', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_discharge_summaries_patient_id'),
            ['patient_id'],
            unique=False,
        )


def downgrade():
    with op.batch_alter_table('discharge_summaries', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_discharge_summaries_patient_id'))
    op.drop_table('discharge_summaries')

    with op.batch_alter_table('referrals', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_referrals_patient_id'))
    op.drop_table('referrals')

    with op.batch_alter_table('appointments', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_appointments_scheduled_start'))
        batch_op.drop_index(batch_op.f('ix_appointments_provider_id'))
        batch_op.drop_index(batch_op.f('ix_appointments_patient_id'))
    op.drop_table('appointments')
