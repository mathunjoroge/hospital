"""add_unified_billing_and_insurance_models

Revision ID: 49d0cf66035c
Revises: b45872b10c9d
Create Date: 2026-09-05 16:34:15.972842

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '49d0cf66035c'
down_revision = 'b45872b10c9d'
branch_labels = None
depends_on = None


def upgrade():
    # Add new billing, insurance, and invoice tables
    op.create_table('invoices',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('invoice_number', sa.String(length=30), nullable=False),
    sa.Column('patient_id', sa.String(length=20), nullable=False),
    sa.Column('status', sa.Enum('DRAFT', 'ISSUED', 'PARTIAL', 'PAID', 'VOID', name='invoicestatus'), nullable=False),
    sa.Column('issued_at', sa.DateTime(), nullable=True),
    sa.Column('due_date', sa.Date(), nullable=True),
    sa.Column('subtotal', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('discount', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('grand_total', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('amount_paid', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('balance', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('insurance_scheme_id', sa.Integer(), nullable=True),
    sa.Column('insurance_claim_ref', sa.String(length=100), nullable=True),
    sa.Column('created_by', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('legacy_source', sa.String(length=30), nullable=True),
    sa.Column('legacy_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('invoices', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_invoices_invoice_number'), ['invoice_number'], unique=True)
        batch_op.create_index(batch_op.f('ix_invoices_patient_id'), ['patient_id'], unique=False)
    op.create_table('invoice_line_items',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('invoice_id', sa.Integer(), nullable=False),
    sa.Column('description', sa.String(length=255), nullable=False),
    sa.Column('category', sa.String(length=50), nullable=False),
    sa.Column('quantity', sa.Numeric(precision=10, scale=3), nullable=False),
    sa.Column('unit_price', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('discount', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('total', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('charge_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['charge_id'], ['charges.id'], ),
    sa.ForeignKeyConstraint(['invoice_id'], ['invoices.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('invoice_line_items', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_invoice_line_items_invoice_id'), ['invoice_id'], unique=False)
    op.create_table('patient_insurance',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('patient_id', sa.String(length=20), nullable=False),
    sa.Column('scheme_id', sa.Integer(), nullable=False),
    sa.Column('member_number', sa.String(length=60), nullable=False),
    sa.Column('principal_member', sa.String(length=100), nullable=True),
    sa.Column('relationship', sa.String(length=30), nullable=True),
    sa.Column('start_date', sa.Date(), nullable=True),
    sa.Column('end_date', sa.Date(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('verified_at', sa.DateTime(), nullable=True),
    sa.Column('verified_by', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ),
    sa.ForeignKeyConstraint(['scheme_id'], ['insurance_schemes.id'], ),
    sa.ForeignKeyConstraint(['verified_by'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('patient_insurance', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_patient_insurance_patient_id'), ['patient_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_patient_insurance_scheme_id'), ['scheme_id'], unique=False)
    op.create_table('insurance_claims',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('claim_number', sa.String(length=40), nullable=False),
    sa.Column('invoice_id', sa.Integer(), nullable=False),
    sa.Column('patient_id', sa.String(length=20), nullable=False),
    sa.Column('scheme_id', sa.Integer(), nullable=False),
    sa.Column('patient_insurance_id', sa.Integer(), nullable=True),
    sa.Column('status', sa.Enum('DRAFT', 'SUBMITTED', 'QUERIED', 'APPROVED', 'REJECTED', 'PAID', 'APPEALED', name='claimstatus'), nullable=False),
    sa.Column('claimed_amount', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('approved_amount', sa.Numeric(precision=12, scale=2), nullable=True),
    sa.Column('co_pay', sa.Numeric(precision=12, scale=2), nullable=True),
    sa.Column('scheme_claim_ref', sa.String(length=100), nullable=True),
    sa.Column('pre_auth_number', sa.String(length=60), nullable=True),
    sa.Column('denial_reason', sa.Text(), nullable=True),
    sa.Column('submitted_at', sa.DateTime(), nullable=True),
    sa.Column('approved_at', sa.DateTime(), nullable=True),
    sa.Column('paid_at', sa.DateTime(), nullable=True),
    sa.Column('appeal_date', sa.DateTime(), nullable=True),
    sa.Column('created_by', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['invoice_id'], ['invoices.id'], ),
    sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ),
    sa.ForeignKeyConstraint(['patient_insurance_id'], ['patient_insurance.id'], ),
    sa.ForeignKeyConstraint(['scheme_id'], ['insurance_schemes.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('insurance_claims', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_insurance_claims_claim_number'), ['claim_number'], unique=True)
        batch_op.create_index(batch_op.f('ix_insurance_claims_invoice_id'), ['invoice_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_insurance_claims_patient_id'), ['patient_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_insurance_claims_scheme_id'), ['scheme_id'], unique=False)
    op.create_table('payments',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('invoice_id', sa.Integer(), nullable=False),
    sa.Column('patient_id', sa.String(length=20), nullable=False),
    sa.Column('amount', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('method', sa.Enum('CASH', 'MPESA', 'INSURANCE', 'BANK', 'WAIVER', 'OTHER', name='paymentmethod'), nullable=False),
    sa.Column('reference', sa.String(length=100), nullable=True),
    sa.Column('receipt_number', sa.String(length=30), nullable=True),
    sa.Column('paid_at', sa.DateTime(), nullable=False),
    sa.Column('recorded_by', sa.Integer(), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('mpesa_checkout_id', sa.String(length=100), nullable=True),
    sa.Column('mpesa_result_code', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['invoice_id'], ['invoices.id'], ),
    sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ),
    sa.ForeignKeyConstraint(['recorded_by'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('receipt_number')
    )
    with op.batch_alter_table('payments', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_payments_invoice_id'), ['invoice_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_payments_patient_id'), ['patient_id'], unique=False)
    op.create_table('insurance_schemes',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('code', sa.String(length=20), nullable=False),
    sa.Column('name', sa.String(length=120), nullable=False),
    sa.Column('scheme_type', sa.String(length=30), nullable=False),
    sa.Column('contact', sa.String(length=100), nullable=True),
    sa.Column('portal_url', sa.String(length=255), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('code')
    )


def downgrade():
    # Drop new tables in reverse dependency order
    op.drop_table('invoices')
    op.drop_table('invoice_line_items')
    op.drop_table('patient_insurance')
    op.drop_table('insurance_claims')
    op.drop_table('payments')
    op.drop_table('insurance_schemes')
