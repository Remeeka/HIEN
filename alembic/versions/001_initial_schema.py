"""initial_schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-10-03 21:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # 1. products
    op.create_table(
        'products',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('sku', sa.String(length=100), nullable=False, unique=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('brand', sa.String(length=100), server_default='HIEN Verified'),
        sa.Column('variant', sa.String(length=100), server_default='Standard Edition'),
        sa.Column('unit_price', sa.Float(), nullable=False),
        sa.Column('selling_price', sa.Float(), nullable=True),
        sa.Column('original_price', sa.Float(), nullable=True),
        sa.Column('weight_kg', sa.Float(), server_default='1.0'),
        sa.Column('hygiene_sensitive', sa.Integer(), server_default='0'),
        sa.Column('fragile', sa.Integer(), server_default='0'),
        sa.Column('base_resale_prob', sa.Float(), server_default='0.85'),
        sa.Column('demand_weight', sa.Float(), server_default='5.0'),
        sa.Column('image_url', sa.Text(), server_default=''),
        sa.Column('description', sa.Text(), server_default=''),
        sa.Column('active', sa.Boolean(), server_default='1'),
        sa.Column('inspection_requirements', sa.JSON(), nullable=True),
        sa.Column('return_eligibility_policy', sa.Text(), server_default='Standard 30-day return policy.'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now())
    )
    op.create_index('idx_product_sku', 'products', ['sku'])
    op.create_index('idx_product_category', 'products', ['category'])

    # 2. locations
    op.create_table(
        'locations',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('location_name', sa.String(length=255), nullable=False),
        sa.Column('city', sa.String(length=100), server_default='Seattle'),
        sa.Column('state', sa.String(length=50), server_default='WA'),
        sa.Column('latitude', sa.Float(), server_default='47.6062'),
        sa.Column('longitude', sa.Float(), server_default='-122.3321'),
        sa.Column('x_km', sa.Float(), server_default='0.0'),
        sa.Column('y_km', sa.Float(), server_default='0.0'),
        sa.Column('zone', sa.String(length=50), server_default='Central'),
        sa.Column('location_type', sa.String(length=50), server_default='store'),
        sa.Column('storage_capacity_units', sa.Integer(), server_default='2500'),
        sa.Column('current_inventory_units', sa.Integer(), server_default='1840'),
        sa.Column('available_storage_units', sa.Integer(), server_default='660'),
        sa.Column('verification_slots_per_day', sa.Integer(), server_default='30'),
        sa.Column('operating_hours', sa.String(length=100), server_default='08:00 - 22:00'),
        sa.Column('delivery_radius_km', sa.Float(), server_default='10.0'),
        sa.Column('hub_enabled', sa.Boolean(), server_default='1'),
        sa.Column('local_delivery_cost_per_km', sa.Float(), server_default='1.25'),
        sa.Column('warehouse_handling_cost_inr', sa.Float(), server_default='48.50'),
        sa.Column('avg_daily_footfall', sa.Integer(), server_default='250'),
        sa.Column('dist_to_warehouse_km', sa.Float(), server_default='420.0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )

    # 3. customers
    op.create_table(
        'customers',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('customer_latitude', sa.Float(), server_default='47.6062'),
        sa.Column('customer_longitude', sa.Float(), server_default='-122.3321'),
        sa.Column('customer_x_km', sa.Float(), server_default='0.0'),
        sa.Column('customer_y_km', sa.Float(), server_default='0.0'),
        sa.Column('home_location_id', sa.String(length=100), nullable=True),
        sa.Column('preferred_delivery_radius_km', sa.Float(), server_default='10.0'),
        sa.Column('customer_segment', sa.String(length=100), server_default='Standard'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )

    # 4. inventory
    op.create_table(
        'inventory',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('sku', sa.String(length=100), sa.ForeignKey('products.sku'), nullable=False),
        sa.Column('product_name', sa.String(length=255), nullable=False),
        sa.Column('category', sa.String(length=100), server_default='General'),
        sa.Column('variant', sa.String(length=100), server_default='Standard'),
        sa.Column('hub_id', sa.String(length=100), sa.ForeignKey('locations.id'), nullable=False),
        sa.Column('hub_name', sa.String(length=255), nullable=False),
        sa.Column('available_sellable', sa.Integer(), server_default='0', nullable=False),
        sa.Column('reserved', sa.Integer(), server_default='0', nullable=False),
        sa.Column('pending_qa', sa.Integer(), server_default='0', nullable=False),
        sa.Column('refurbishment', sa.Integer(), server_default='0', nullable=False),
        sa.Column('quarantined', sa.Integer(), server_default='0', nullable=False),
        sa.Column('damaged', sa.Integer(), server_default='0', nullable=False),
        sa.Column('in_transit', sa.Integer(), server_default='0', nullable=False),
        sa.Column('total_stock', sa.Integer(), server_default='0', nullable=False),
        sa.Column('target_stock', sa.Integer(), server_default='12'),
        sa.Column('forecast_demand', sa.Integer(), server_default='8'),
        sa.Column('disposition', sa.String(length=50), server_default='sellable'),
        sa.Column('storage_location', sa.String(length=100), server_default='Bay A'),
        sa.Column('unit_cost', sa.Float(), server_default='50.0'),
        sa.Column('selling_price', sa.Float(), server_default='100.0'),
        sa.Column('last_updated', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint('sku', 'hub_id', name='uq_inventory_sku_hub'),
        sa.CheckConstraint('available_sellable >= 0', name='chk_available_nonneg'),
        sa.CheckConstraint('reserved >= 0', name='chk_reserved_nonneg'),
        sa.CheckConstraint('pending_qa >= 0', name='chk_pending_qa_nonneg')
    )
    op.create_index('idx_inv_sku_hub', 'inventory', ['sku', 'hub_id'])

    # 5. inventory_transactions
    op.create_table(
        'inventory_transactions',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('sku', sa.String(length=100), nullable=False),
        sa.Column('product_name', sa.String(length=255), server_default=''),
        sa.Column('hub_id', sa.String(length=100), nullable=False),
        sa.Column('source_hub_id', sa.String(length=100), nullable=True),
        sa.Column('destination_hub_id', sa.String(length=100), nullable=True),
        sa.Column('quantity_delta', sa.Integer(), nullable=False),
        sa.Column('from_state', sa.String(length=50), nullable=False),
        sa.Column('to_state', sa.String(length=50), nullable=False),
        sa.Column('reason', sa.String(length=100), nullable=False),
        sa.Column('reference_id', sa.String(length=100), nullable=False),
        sa.Column('actor', sa.String(length=100), server_default='System'),
        sa.Column('idempotency_key', sa.String(length=150), unique=True, nullable=True),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )
    op.create_index('idx_tx_ref_sku', 'inventory_transactions', ['reference_id', 'sku'])
    op.create_index('idx_tx_timestamp', 'inventory_transactions', ['timestamp'])

    # 6. audit_ledger
    op.create_table(
        'audit_ledger',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('actor', sa.String(length=100), nullable=False),
        sa.Column('role', sa.String(length=100), server_default='User'),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('entity_type', sa.String(length=100), nullable=False),
        sa.Column('entity_id', sa.String(length=100), nullable=False),
        sa.Column('previous_state', sa.Text(), server_default=''),
        sa.Column('new_state', sa.Text(), server_default=''),
        sa.Column('result', sa.String(length=50), server_default='SUCCESS'),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )
    op.create_index('idx_audit_entity', 'audit_ledger', ['entity_type', 'entity_id'])
    op.create_index('idx_audit_timestamp', 'audit_ledger', ['timestamp'])

    # 7. returns
    op.create_table(
        'returns',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('rma_number', sa.String(length=100), unique=True, nullable=False),
        sa.Column('order_reference', sa.String(length=100), server_default='ORD-UNKNOWN'),
        sa.Column('sku', sa.String(length=100), sa.ForeignKey('products.sku'), nullable=False),
        sa.Column('product_name', sa.String(length=255), nullable=False),
        sa.Column('category', sa.String(length=100), server_default='General'),
        sa.Column('quantity', sa.Integer(), server_default='1'),
        sa.Column('hub_id', sa.String(length=100), sa.ForeignKey('locations.id'), nullable=False),
        sa.Column('hub_name', sa.String(length=255), nullable=False),
        sa.Column('return_reason', sa.Text(), server_default=''),
        sa.Column('reported_condition', sa.String(length=50), server_default='like_new'),
        sa.Column('packaging_condition', sa.String(length=50), server_default='opened_good'),
        sa.Column('accessory_completeness', sa.String(length=50), server_default='complete'),
        sa.Column('status', sa.String(length=50), server_default='registered'),
        sa.Column('primary_image', sa.Text(), server_default=''),
        sa.Column('images_json', sa.JSON(), nullable=True),
        sa.Column('inventory_restocked_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now())
    )
    op.create_index('idx_returns_rma', 'returns', ['rma_number'])
    op.create_index('idx_returns_sku', 'returns', ['sku'])

    # 8. return_assessments
    op.create_table(
        'return_assessments',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('return_id', sa.String(length=100), sa.ForeignKey('returns.id'), unique=True, nullable=False),
        sa.Column('predicted_disposition', sa.String(length=50), nullable=False),
        sa.Column('confidence_score', sa.Float(), server_default='0.85'),
        sa.Column('class_probabilities', sa.JSON(), nullable=True),
        sa.Column('cnn_model', sa.String(length=100), server_default='CNN Condition Classifier'),
        sa.Column('cnn_condition', sa.String(length=50), server_default='Like_New'),
        sa.Column('cnn_confidence', sa.Float(), server_default='0.94'),
        sa.Column('defect_labels', sa.JSON(), nullable=True),
        sa.Column('risk_flags', sa.JSON(), nullable=True),
        sa.Column('missing_features', sa.JSON(), nullable=True),
        sa.Column('recommended_next_step', sa.Text(), server_default=''),
        sa.Column('model_version', sa.String(length=100), server_default='RandomForest-v1.0 / CNN-v1.0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )

    # 9. physical_inspections
    op.create_table(
        'physical_inspections',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('return_id', sa.String(length=100), sa.ForeignKey('returns.id'), unique=True, nullable=False),
        sa.Column('inspector_id', sa.String(length=100), server_default='u-2'),
        sa.Column('inspector_name', sa.String(length=100), server_default='Marcus Vance'),
        sa.Column('checks_json', sa.JSON(), nullable=True),
        sa.Column('final_disposition', sa.String(length=50), nullable=False),
        sa.Column('inspector_notes', sa.Text(), server_default=''),
        sa.Column('evidence_image_url', sa.Text(), server_default=''),
        sa.Column('completed_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )

    # 10. orders
    op.create_table(
        'orders',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('customer_id', sa.String(length=100), sa.ForeignKey('customers.id'), nullable=True),
        sa.Column('order_timestamp', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('sku', sa.String(length=100), nullable=False),
        sa.Column('product_name', sa.String(length=255), server_default=''),
        sa.Column('quantity', sa.Integer(), server_default='1'),
        sa.Column('assigned_hub_id', sa.String(length=100), sa.ForeignKey('locations.id'), nullable=False),
        sa.Column('hub_name', sa.String(length=255), server_default=''),
        sa.Column('unit_price', sa.Float(), server_default='99.99'),
        sa.Column('weight_kg', sa.Float(), server_default='1.0'),
        sa.Column('category', sa.String(length=100), server_default='General'),
        sa.Column('status', sa.String(length=50), server_default='pending'),
        sa.Column('delivery_type', sa.String(length=50), server_default='store_pickup'),
        sa.Column('promised_delivery', sa.String(length=100), server_default='Same-Day 18:00'),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )

    # 11. transfers
    op.create_table(
        'transfers',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('transfer_code', sa.String(length=100), unique=True, nullable=False),
        sa.Column('source_hub_id', sa.String(length=100), sa.ForeignKey('locations.id'), nullable=False),
        sa.Column('source_hub_name', sa.String(length=255), nullable=False),
        sa.Column('dest_hub_id', sa.String(length=100), sa.ForeignKey('locations.id'), nullable=False),
        sa.Column('dest_hub_name', sa.String(length=255), nullable=False),
        sa.Column('sku', sa.String(length=100), nullable=False),
        sa.Column('product_name', sa.String(length=255), nullable=False),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=50), server_default='proposed'),
        sa.Column('transport_method', sa.String(length=100), server_default='Local Green Courier Fleet'),
        sa.Column('carrier_name', sa.String(length=100), server_default='TBD'),
        sa.Column('tracking_number', sa.String(length=100), server_default='Unassigned'),
        sa.Column('distance_km', sa.Float(), server_default='52.4'),
        sa.Column('transport_cost', sa.Float(), server_default='65.50'),
        sa.Column('emissions_kg', sa.Float(), server_default='6.8'),
        sa.Column('net_benefit', sa.Float(), server_default='120.0'),
        sa.Column('planned_departure', sa.DateTime(), nullable=True),
        sa.Column('actual_departure', sa.DateTime(), nullable=True),
        sa.Column('estimated_arrival', sa.DateTime(), nullable=True),
        sa.Column('actual_arrival', sa.DateTime(), nullable=True),
        sa.Column('receipt_details', sa.JSON(), nullable=True),
        sa.Column('events_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )

    # 12. transfer_discrepancies
    op.create_table(
        'transfer_discrepancies',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('transfer_id', sa.String(length=100), sa.ForeignKey('transfers.id'), nullable=False),
        sa.Column('expected_quantity', sa.Integer(), nullable=False),
        sa.Column('received_quantity', sa.Integer(), nullable=False),
        sa.Column('damaged_quantity', sa.Integer(), server_default='0'),
        sa.Column('missing_quantity', sa.Integer(), server_default='0'),
        sa.Column('notes', sa.Text(), server_default=''),
        sa.Column('inspector_name', sa.String(length=100), server_default='Marcus Vance'),
        sa.Column('status', sa.String(length=50), server_default='recorded'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )

    # 13. demand_forecasts
    op.create_table(
        'demand_forecasts',
        sa.Column('id', sa.String(length=100), primary_key=True),
        sa.Column('sku', sa.String(length=100), nullable=False),
        sa.Column('product_name', sa.String(length=255), server_default=''),
        sa.Column('hub_id', sa.String(length=100), sa.ForeignKey('locations.id'), nullable=False),
        sa.Column('hub_name', sa.String(length=255), server_default=''),
        sa.Column('forecast_date', sa.String(length=50), nullable=True),
        sa.Column('horizon_days', sa.Integer(), server_default='7'),
        sa.Column('predicted_demand', sa.Float(), server_default='0.0'),
        sa.Column('historical_demand_7d', sa.Float(), server_default='0.0'),
        sa.Column('historical_demand_14d', sa.Float(), server_default='0.0'),
        sa.Column('historical_demand_30d', sa.Float(), server_default='0.0'),
        sa.Column('demand_score', sa.Float(), server_default='50.0'),
        sa.Column('demand_tier', sa.String(length=50), server_default='Moderate Demand'),
        sa.Column('model_name', sa.String(length=100), server_default='HistGradientBoostingRegressor'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )
    op.create_index('idx_forecast_sku_hub_date', 'demand_forecasts', ['sku', 'hub_id', 'forecast_date'])

    # 14. network_policy_config
    op.create_table(
        'network_policy_config',
        sa.Column('id', sa.Integer(), primary_key=True, default=1),
        sa.Column('planning_horizon_days', sa.Integer(), server_default='14'),
        sa.Column('min_expected_transfer_benefit', sa.Float(), server_default='35.0'),
        sa.Column('transport_cost_per_km', sa.Float(), server_default='1.25'),
        sa.Column('emissions_factor_kg_per_km', sa.Float(), server_default='0.192'),
        sa.Column('hub_capacity_threshold_alert', sa.Float(), server_default='85.0'),
        sa.Column('ai_confidence_threshold', sa.Float(), server_default='75.0'),
        sa.Column('baseline_central_cost_per_unit', sa.Float(), server_default='48.50'),
        sa.Column('baseline_central_distance_km', sa.Float(), server_default='420.0'),
        sa.Column('updated_by', sa.String(length=100), server_default='System Administrator'),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now())
    )

def downgrade() -> None:
    op.drop_table('network_policy_config')
    op.drop_table('demand_forecasts')
    op.drop_table('transfer_discrepancies')
    op.drop_table('transfers')
    op.drop_table('orders')
    op.drop_table('physical_inspections')
    op.drop_table('return_assessments')
    op.drop_table('returns')
    op.drop_table('audit_ledger')
    op.drop_table('inventory_transactions')
    op.drop_table('inventory')
    op.drop_table('customers')
    op.drop_table('locations')
    op.drop_table('products')
