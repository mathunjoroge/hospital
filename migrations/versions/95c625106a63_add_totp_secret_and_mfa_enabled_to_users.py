"""add totp_secret and mfa_enabled to users

Revision ID: 95c625106a63
Revises: bb45ec7f300c
Create Date: 2026-09-05 16:11:15.454843

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '95c625106a63'
down_revision = 'bb45ec7f300c'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    user_cols = [c['name'] for c in inspector.get_columns('users')]
    if 'totp_secret' not in user_cols:
        op.add_column('users', sa.Column('totp_secret', sa.String(length=64), nullable=True))
    if 'mfa_enabled' not in user_cols:
        op.add_column('users', sa.Column('mfa_enabled', sa.Boolean(), nullable=False, server_default=sa.text('0')))


def downgrade():
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    user_cols = [c['name'] for c in inspector.get_columns('users')]
    if 'mfa_enabled' in user_cols:
        op.drop_column('users', 'mfa_enabled')
    if 'totp_secret' in user_cols:
        op.drop_column('users', 'totp_secret')
