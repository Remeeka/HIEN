"""add_users_table

Revision ID: 002_add_users_table
Revises: 001_initial_schema
Create Date: 2026-10-04 02:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '002_add_users_table'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('username', sa.String(length=100), nullable=False, unique=True),
        sa.Column('email', sa.String(length=255), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), server_default=''),
        sa.Column('role', sa.String(length=50), server_default='operator'),
        sa.Column('assigned_hub_id', sa.String(length=100), sa.ForeignKey('locations.id'), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default='1'),
        sa.Column('avatar_url', sa.String(length=500), server_default=''),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now())
    )
    op.create_index('idx_user_username', 'users', ['username'])
    op.create_index('idx_user_email', 'users', ['email'])

def downgrade() -> None:
    op.drop_index('idx_user_email', table_name='users')
    op.drop_index('idx_user_username', table_name='users')
    op.drop_table('users')
