"""add_facility_stock_movement_and_po_sod_columns

Revision ID: 3d50aaf51986
Revises: c3f8a1d92e74
Create Date: 2026-09-06 16:57:09.568659

"""
from alembic import op
import sqlalchemy as sa
import departments.crypto


# revision identifiers, used by Alembic.
revision = '3d50aaf51986'
down_revision = 'c3f8a1d92e74'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if 'facilities' not in tables:
        op.create_table('facilities',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('facility_code', sa.String(length=50), nullable=True),
        sa.Column('facility_type', sa.String(length=50), nullable=False),
        sa.Column('address', sa.String(length=255), nullable=True),
        sa.Column('contact_phone', sa.String(length=30), nullable=True),
        sa.Column('contact_email', sa.String(length=120), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_self', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('facilities', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_facilities_facility_code'), ['facility_code'], unique=True)
            batch_op.create_index(batch_op.f('ix_facilities_is_self'), ['is_self'], unique=False)
            batch_op.create_index(batch_op.f('ix_facilities_name'), ['name'], unique=False)

    if 'suppliers' not in tables:
        op.create_table('suppliers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('contact_email', sa.String(length=120), nullable=True),
        sa.Column('phone', sa.String(length=30), nullable=True),
        sa.Column('address', sa.String(length=255), nullable=True),
        sa.Column('lead_time_days', sa.Integer(), nullable=False, server_default='3'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
        )

    if 'purchase_orders' not in tables:
        op.create_table('purchase_orders',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('po_number', sa.String(length=50), nullable=False),
        sa.Column('supplier_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='DRAFT'),
        sa.Column('created_by_id', sa.Integer(), nullable=True),
        sa.Column('approved_by_id', sa.Integer(), nullable=True),
        sa.Column('received_by_id', sa.Integer(), nullable=True),
        sa.Column('sod_warning', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('total_cost', sa.Numeric(precision=10, scale=2), nullable=False, server_default='0.00'),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('ordered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('received_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['supplier_id'], ['suppliers.id'], ),
        sa.ForeignKeyConstraint(['created_by_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['approved_by_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['received_by_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('po_number')
        )
    else:
        po_cols = [c['name'] for c in inspector.get_columns('purchase_orders')]
        with op.batch_alter_table('purchase_orders', schema=None) as batch_op:
            if 'created_by_id' not in po_cols:
                batch_op.add_column(sa.Column('created_by_id', sa.Integer(), nullable=True))
                batch_op.create_foreign_key('fk_po_created_by', 'users', ['created_by_id'], ['id'])
            if 'approved_by_id' not in po_cols:
                batch_op.add_column(sa.Column('approved_by_id', sa.Integer(), nullable=True))
                batch_op.create_foreign_key('fk_po_approved_by', 'users', ['approved_by_id'], ['id'])
            if 'received_by_id' not in po_cols:
                batch_op.add_column(sa.Column('received_by_id', sa.Integer(), nullable=True))
                batch_op.create_foreign_key('fk_po_received_by', 'users', ['received_by_id'], ['id'])
            if 'sod_warning' not in po_cols:
                batch_op.add_column(sa.Column('sod_warning', sa.Boolean(), nullable=False, server_default='0'))

    if 'purchase_order_items' not in tables:
        op.create_table('purchase_order_items',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('po_id', sa.Integer(), nullable=False),
        sa.Column('item_type', sa.String(length=20), nullable=False, server_default='DRUG'),
        sa.Column('drug_id', sa.Integer(), nullable=True),
        sa.Column('non_pharm_item_id', sa.Integer(), nullable=True),
        sa.Column('vote_head_id', sa.Integer(), nullable=True),
        sa.Column('quantity_ordered', sa.Integer(), nullable=False),
        sa.Column('unit_cost', sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column('quantity_received', sa.Integer(), nullable=False, server_default='0'),
        sa.ForeignKeyConstraint(['po_id'], ['purchase_orders.id'], ),
        sa.ForeignKeyConstraint(['drug_id'], ['drugs.id'], ),
        sa.PrimaryKeyConstraint('id')
        )
    else:
        poi_cols = [c['name'] for c in inspector.get_columns('purchase_order_items')]
        with op.batch_alter_table('purchase_order_items', schema=None) as batch_op:
            if 'item_type' not in poi_cols:
                batch_op.add_column(sa.Column('item_type', sa.String(length=20), nullable=False, server_default='DRUG'))
            if 'non_pharm_item_id' not in poi_cols:
                batch_op.add_column(sa.Column('non_pharm_item_id', sa.Integer(), nullable=True))

    if 'stock_movements' not in tables:
        op.create_table('stock_movements',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('item_type', sa.String(length=20), nullable=False),
        sa.Column('item_id', sa.Integer(), nullable=False),
        sa.Column('batch_id', sa.Integer(), nullable=True),
        sa.Column('movement_type', sa.String(length=30), nullable=False),
        sa.Column('quantity_delta', sa.Integer(), nullable=False),
        sa.Column('balance_after', sa.Integer(), nullable=False),
        sa.Column('reference_type', sa.String(length=50), nullable=True),
        sa.Column('reference_id', sa.String(length=50), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('facility_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['batch_id'], ['batches.id'], ),
        sa.ForeignKeyConstraint(['facility_id'], ['facilities.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
        )
        with op.batch_alter_table('stock_movements', schema=None) as batch_op:
            batch_op.create_index('idx_sm_item', ['item_type', 'item_id'], unique=False)
            batch_op.create_index('idx_sm_ref', ['reference_type', 'reference_id'], unique=False)
            batch_op.create_index(batch_op.f('ix_stock_movements_created_at'), ['created_at'], unique=False)
            batch_op.create_index(batch_op.f('ix_stock_movements_item_id'), ['item_id'], unique=False)
            batch_op.create_index(batch_op.f('ix_stock_movements_item_type'), ['item_type'], unique=False)
            batch_op.create_index(batch_op.f('ix_stock_movements_movement_type'), ['movement_type'], unique=False)


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()

    if 'stock_movements' in tables:
        op.drop_table('stock_movements')
    if 'purchase_order_items' in tables:
        op.drop_table('purchase_order_items')
    if 'purchase_orders' in tables:
        op.drop_table('purchase_orders')
    if 'suppliers' in tables:
        op.drop_table('suppliers')
    if 'facilities' in tables:
        op.drop_table('facilities')
