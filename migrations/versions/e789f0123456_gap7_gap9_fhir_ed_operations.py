"""gap7_gap9_fhir_ed_operations

Revision ID: e789f0123456
Revises: d1326ffcbc3f
Create Date: 2026-09-13 23:35:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e789f0123456'
down_revision = 'd1326ffcbc3f'
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table('triage_assessments', schema=None) as batch_op:
        batch_op.add_column(sa.Column('arrival_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('triage_completed_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('seen_by_doctor_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('disposition_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('disposition', sa.String(length=30), nullable=True))
        batch_op.add_column(sa.Column('bed_assigned_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('bed_label', sa.String(length=20), nullable=True))
        batch_op.add_column(sa.Column('re_evaluation_due_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('last_re_evaluation_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('re_evaluation_notes', sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table('triage_assessments', schema=None) as batch_op:
        batch_op.drop_column('re_evaluation_notes')
        batch_op.drop_column('last_re_evaluation_at')
        batch_op.drop_column('re_evaluation_due_at')
        batch_op.drop_column('bed_label')
        batch_op.drop_column('bed_assigned_at')
        batch_op.drop_column('disposition')
        batch_op.drop_column('disposition_at')
        batch_op.drop_column('seen_by_doctor_at')
        batch_op.drop_column('triage_completed_at')
        batch_op.drop_column('arrival_at')
