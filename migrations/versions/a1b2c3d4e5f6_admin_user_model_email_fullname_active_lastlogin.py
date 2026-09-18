"""admin: add email, full_name, is_active, created_at, last_login to users table

Revision ID: a1b2c3d4e5f6
Revises: fefcbf381ba9
Create Date: 2026-09-18 00:00:00.000000

Adds five columns that were missing from the staff User model:
- email        : unique contact email for password-reset notifications
- full_name    : display name for the UI (manage_users table)
- is_active    : soft-disable without deleting accounts
- created_at   : account creation timestamp for audit purposes
- last_login   : populated on successful login for the system overview
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = 'f9c69a8e3e2e'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = [c["name"] for c in inspector.get_columns("users")]
    with op.batch_alter_table('users', schema=None) as batch_op:
        if 'email' not in cols:
            batch_op.add_column(sa.Column('email', sa.String(length=120), nullable=True))
            batch_op.create_unique_constraint('uq_users_email', ['email'])
            batch_op.create_index('ix_users_email', ['email'], unique=True)
        if 'full_name' not in cols:
            batch_op.add_column(sa.Column('full_name', sa.String(length=120), nullable=True))
        if 'is_active' not in cols:
            batch_op.add_column(
                sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true())
            )
        if 'created_at' not in cols:
            batch_op.add_column(
                sa.Column('created_at', sa.DateTime(timezone=True), nullable=True)
            )
        if 'last_login' not in cols:
            batch_op.add_column(
                sa.Column('last_login', sa.DateTime(timezone=True), nullable=True)
            )


def downgrade():
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_index('ix_users_email')
        batch_op.drop_constraint('uq_users_email', type_='unique')
        batch_op.drop_column('last_login')
        batch_op.drop_column('created_at')
        batch_op.drop_column('is_active')
        batch_op.drop_column('full_name')
        batch_op.drop_column('email')
