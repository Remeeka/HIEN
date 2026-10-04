from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, DateTime, JSON, ForeignKey,
    UniqueConstraint, CheckConstraint, Index
)
from sqlalchemy.orm import relationship
from backend.db.database import Base

class Product(Base):
    __tablename__ = "products"

    id = Column(String(100), primary_key=True, index=True) # e.g. SKU-AUD-801
    sku = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False, index=True)
    brand = Column(String(100), default="HIEN Verified")
    variant = Column(String(100), default="Standard Edition")
    unit_price = Column(Float, nullable=False)
    selling_price = Column(Float, nullable=True)
    original_price = Column(Float, nullable=True)
    weight_kg = Column(Float, default=1.0)
    hygiene_sensitive = Column(Integer, default=0)
    fragile = Column(Integer, default=0)
    base_resale_prob = Column(Float, default=0.85)
    demand_weight = Column(Float, default=5.0)
    image_url = Column(Text, default="")
    description = Column(Text, default="")
    active = Column(Boolean, default=True)
    inspection_requirements = Column(JSON, default=list)
    return_eligibility_policy = Column(Text, default="Standard 30-day return policy.")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    inventory_records = relationship("Inventory", back_populates="product", cascade="all, delete-orphan")


class Location(Base):
    __tablename__ = "locations"

    id = Column(String(100), primary_key=True, index=True) # e.g. hub-sea-01 or LOC001
    location_name = Column(String(255), nullable=False)
    city = Column(String(100), default="Seattle")
    state = Column(String(50), default="WA")
    latitude = Column(Float, default=47.6062)
    longitude = Column(Float, default=-122.3321)
    x_km = Column(Float, default=0.0)
    y_km = Column(Float, default=0.0)
    zone = Column(String(50), default="Central")
    location_type = Column(String(50), default="store") # store, access_point, warehouse
    storage_capacity_units = Column(Integer, default=2500)
    current_inventory_units = Column(Integer, default=1840)
    available_storage_units = Column(Integer, default=660)
    verification_slots_per_day = Column(Integer, default=30)
    operating_hours = Column(String(100), default="08:00 - 22:00")
    delivery_radius_km = Column(Float, default=10.0)
    hub_enabled = Column(Boolean, default=True)
    local_delivery_cost_per_km = Column(Float, default=1.25)
    warehouse_handling_cost_inr = Column(Float, default=48.50)
    avg_daily_footfall = Column(Integer, default=250)
    dist_to_warehouse_km = Column(Float, default=420.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    inventory_records = relationship("Inventory", back_populates="hub", cascade="all, delete-orphan")


class Customer(Base):
    __tablename__ = "customers"

    id = Column(String(100), primary_key=True, index=True)
    customer_latitude = Column(Float, default=47.6062)
    customer_longitude = Column(Float, default=-122.3321)
    customer_x_km = Column(Float, default=0.0)
    customer_y_km = Column(Float, default=0.0)
    home_location_id = Column(String(100), nullable=True)
    preferred_delivery_radius_km = Column(Float, default=10.0)
    customer_segment = Column(String(100), default="Standard")
    created_at = Column(DateTime, default=datetime.utcnow)


class Inventory(Base):
    __tablename__ = "inventory"

    id = Column(String(100), primary_key=True, index=True)
    sku = Column(String(100), ForeignKey("products.sku"), nullable=False, index=True)
    product_name = Column(String(255), nullable=False)
    category = Column(String(100), default="General")
    variant = Column(String(100), default="Standard")
    hub_id = Column(String(100), ForeignKey("locations.id"), nullable=False, index=True)
    hub_name = Column(String(255), nullable=False)
    
    # Strictly managed discrete inventory buckets
    available_sellable = Column(Integer, default=0, nullable=False)
    reserved = Column(Integer, default=0, nullable=False)
    pending_qa = Column(Integer, default=0, nullable=False)
    refurbishment = Column(Integer, default=0, nullable=False)
    quarantined = Column(Integer, default=0, nullable=False)
    damaged = Column(Integer, default=0, nullable=False)
    in_transit = Column(Integer, default=0, nullable=False)
    total_stock = Column(Integer, default=0, nullable=False)

    target_stock = Column(Integer, default=12)
    forecast_demand = Column(Integer, default=8)
    disposition = Column(String(50), default="sellable") # sellable, quarantined, refurbishment
    storage_location = Column(String(100), default="Aisle 1, Bay A")
    unit_cost = Column(Float, default=50.0)
    selling_price = Column(Float, default=100.0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Constraints & Indexes
    __table_args__ = (
        UniqueConstraint("sku", "hub_id", name="uq_inventory_sku_hub"),
        CheckConstraint("available_sellable >= 0", name="chk_available_nonneg"),
        CheckConstraint("reserved >= 0", name="chk_reserved_nonneg"),
        CheckConstraint("pending_qa >= 0", name="chk_pending_qa_nonneg"),
        CheckConstraint("refurbishment >= 0", name="chk_refurbishment_nonneg"),
        CheckConstraint("quarantined >= 0", name="chk_quarantined_nonneg"),
        CheckConstraint("in_transit >= 0", name="chk_in_transit_nonneg"),
        Index("idx_inv_sku_hub", "sku", "hub_id"),
    )

    # Relationships
    product = relationship("Product", back_populates="inventory_records")
    hub = relationship("Location", back_populates="inventory_records")


class InventoryTransaction(Base):
    __tablename__ = "inventory_transactions"

    id = Column(String(100), primary_key=True, index=True) # e.g. tx-1791043000
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    sku = Column(String(100), nullable=False, index=True)
    product_name = Column(String(255), default="")
    hub_id = Column(String(100), nullable=False, index=True)
    source_hub_id = Column(String(100), nullable=True)
    destination_hub_id = Column(String(100), nullable=True)
    quantity_delta = Column(Integer, nullable=False)
    from_state = Column(String(50), nullable=False)
    to_state = Column(String(50), nullable=False)
    reason = Column(String(100), nullable=False) # RETURN_REGISTERED, RETURN_RESTOCK, ORDER_RESERVED, etc.
    reference_id = Column(String(100), nullable=False, index=True) # RMA-xxx, ORD-xxx, TR-xxx
    actor = Column(String(100), default="System")
    idempotency_key = Column(String(150), unique=True, nullable=True)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_tx_ref_sku", "reference_id", "sku"),
        Index("idx_tx_timestamp", "timestamp"),
    )


class AuditLedger(Base):
    __tablename__ = "audit_ledger"

    id = Column(String(100), primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    actor = Column(String(100), nullable=False)
    role = Column(String(100), default="User")
    action = Column(String(100), nullable=False, index=True)
    entity_type = Column(String(100), nullable=False)
    entity_id = Column(String(100), nullable=False, index=True)
    previous_state = Column(Text, default="")
    new_state = Column(Text, default="")
    result = Column(String(50), default="SUCCESS") # SUCCESS, FAILURE, BLOCKED
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_audit_entity", "entity_type", "entity_id"),
        Index("idx_audit_timestamp", "timestamp"),
    )


class Return(Base):
    __tablename__ = "returns"

    id = Column(String(100), primary_key=True, index=True) # e.g. RET000001 or RMA-2026-9081
    rma_number = Column(String(100), unique=True, nullable=False, index=True)
    order_reference = Column(String(100), default="ORD-UNKNOWN")
    sku = Column(String(100), ForeignKey("products.sku"), nullable=False, index=True)
    product_name = Column(String(255), nullable=False)
    category = Column(String(100), default="General")
    quantity = Column(Integer, default=1)
    hub_id = Column(String(100), ForeignKey("locations.id"), nullable=False, index=True)
    hub_name = Column(String(255), nullable=False)
    return_reason = Column(Text, default="")
    reported_condition = Column(String(50), default="like_new")
    packaging_condition = Column(String(50), default="opened_good")
    accessory_completeness = Column(String(50), default="complete")
    status = Column(String(50), default="registered", index=True) # registered, assessment_pending, inspected, approved_resale, refurbishment_required, quarantined, not_suitable
    primary_image = Column(Text, default="")
    images_json = Column(JSON, default=list)
    inventory_restocked_at = Column(DateTime, nullable=True) # Idempotency timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    assessment = relationship("ReturnAssessment", back_populates="return_item", uselist=False, cascade="all, delete-orphan")
    inspection = relationship("PhysicalInspection", back_populates="return_item", uselist=False, cascade="all, delete-orphan")


class ReturnAssessment(Base):
    __tablename__ = "return_assessments"

    id = Column(String(100), primary_key=True, index=True)
    return_id = Column(String(100), ForeignKey("returns.id"), unique=True, nullable=False)
    predicted_disposition = Column(String(50), nullable=False) # approved_resale, refurbishment_required, quarantined, not_suitable
    confidence_score = Column(Float, default=0.85)
    class_probabilities = Column(JSON, default=dict)
    cnn_model = Column(String(100), default="CNN Condition Classifier")
    cnn_condition = Column(String(50), default="Like_New")
    cnn_confidence = Column(Float, default=0.94)
    defect_labels = Column(JSON, default=list)
    risk_flags = Column(JSON, default=list)
    missing_features = Column(JSON, default=list)
    recommended_next_step = Column(Text, default="")
    model_version = Column(String(100), default="RandomForest-v1.0 / CNN-v1.0")
    created_at = Column(DateTime, default=datetime.utcnow)

    return_item = relationship("Return", back_populates="assessment")


class PhysicalInspection(Base):
    __tablename__ = "physical_inspections"

    id = Column(String(100), primary_key=True, index=True)
    return_id = Column(String(100), ForeignKey("returns.id"), unique=True, nullable=False)
    inspector_id = Column(String(100), default="u-2")
    inspector_name = Column(String(100), default="Marcus Vance")
    checks_json = Column(JSON, default=list)
    final_disposition = Column(String(50), nullable=False)
    inspector_notes = Column(Text, default="")
    evidence_image_url = Column(Text, default="")
    completed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    return_item = relationship("Return", back_populates="inspection")


class Order(Base):
    __tablename__ = "orders"

    id = Column(String(100), primary_key=True, index=True) # e.g. ORD-WA-8812
    customer_id = Column(String(100), ForeignKey("customers.id"), nullable=True)
    order_timestamp = Column(DateTime, default=datetime.utcnow)
    sku = Column(String(100), nullable=False, index=True)
    product_name = Column(String(255), default="")
    quantity = Column(Integer, default=1)
    assigned_hub_id = Column(String(100), ForeignKey("locations.id"), nullable=False, index=True)
    hub_name = Column(String(255), default="")
    unit_price = Column(Float, default=99.99)
    weight_kg = Column(Float, default=1.0)
    category = Column(String(100), default="General")
    status = Column(String(50), default="pending", index=True) # pending, reserved, prepared, ready_for_pickup, dispatched, fulfilled, cancelled
    delivery_type = Column(String(50), default="store_pickup")
    promised_delivery = Column(String(100), default="Same-Day 18:00")
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Transfer(Base):
    __tablename__ = "transfers"

    id = Column(String(100), primary_key=True, index=True) # e.g. TR-2026-4182
    transfer_code = Column(String(100), unique=True, nullable=False, index=True)
    source_hub_id = Column(String(100), ForeignKey("locations.id"), nullable=False, index=True)
    source_hub_name = Column(String(255), nullable=False)
    dest_hub_id = Column(String(100), ForeignKey("locations.id"), nullable=False, index=True)
    dest_hub_name = Column(String(255), nullable=False)
    sku = Column(String(100), nullable=False, index=True)
    product_name = Column(String(255), nullable=False)
    quantity = Column(Integer, nullable=False)
    status = Column(String(50), default="proposed", index=True) # proposed, approved, in_transit, arrived, received, exception, cancelled
    transport_method = Column(String(100), default="Local Green Courier Fleet")
    carrier_name = Column(String(100), default="TBD")
    tracking_number = Column(String(100), default="Unassigned")
    distance_km = Column(Float, default=52.4)
    transport_cost = Column(Float, default=65.50)
    emissions_kg = Column(Float, default=6.8)
    net_benefit = Column(Float, default=120.0)
    planned_departure = Column(DateTime, nullable=True)
    actual_departure = Column(DateTime, nullable=True)
    estimated_arrival = Column(DateTime, nullable=True)
    actual_arrival = Column(DateTime, nullable=True)
    receipt_details = Column(JSON, default=dict)
    events_json = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)

    discrepancies = relationship("TransferDiscrepancy", back_populates="transfer", cascade="all, delete-orphan")


class TransferDiscrepancy(Base):
    __tablename__ = "transfer_discrepancies"

    id = Column(String(100), primary_key=True, index=True)
    transfer_id = Column(String(100), ForeignKey("transfers.id"), nullable=False)
    expected_quantity = Column(Integer, nullable=False)
    received_quantity = Column(Integer, nullable=False)
    damaged_quantity = Column(Integer, default=0)
    missing_quantity = Column(Integer, default=0)
    notes = Column(Text, default="")
    inspector_name = Column(String(100), default="Marcus Vance")
    status = Column(String(50), default="recorded")
    created_at = Column(DateTime, default=datetime.utcnow)

    transfer = relationship("Transfer", back_populates="discrepancies")


class DemandForecast(Base):
    __tablename__ = "demand_forecasts"

    id = Column(String(100), primary_key=True, index=True)
    sku = Column(String(100), nullable=False, index=True)
    product_name = Column(String(255), default="")
    hub_id = Column(String(100), ForeignKey("locations.id"), nullable=False, index=True)
    hub_name = Column(String(255), default="")
    forecast_date = Column(String(50), index=True)
    horizon_days = Column(Integer, default=7)
    predicted_demand = Column(Float, default=0.0)
    historical_demand_7d = Column(Float, default=0.0)
    historical_demand_14d = Column(Float, default=0.0)
    historical_demand_30d = Column(Float, default=0.0)
    demand_score = Column(Float, default=50.0)
    demand_tier = Column(String(50), default="Moderate Demand")
    model_name = Column(String(100), default="HistGradientBoostingRegressor")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_forecast_sku_hub_date", "sku", "hub_id", "forecast_date"),
    )


class NetworkPolicyConfig(Base):
    __tablename__ = "network_policy_config"

    id = Column(Integer, primary_key=True, default=1)
    planning_horizon_days = Column(Integer, default=14)
    min_expected_transfer_benefit = Column(Float, default=35.0)
    transport_cost_per_km = Column(Float, default=1.25)
    emissions_factor_kg_per_km = Column(Float, default=0.192)
    hub_capacity_threshold_alert = Column(Float, default=85.0)
    ai_confidence_threshold = Column(Float, default=75.0)
    baseline_central_cost_per_unit = Column(Float, default=48.50)
    baseline_central_distance_km = Column(Float, default=420.0)
    updated_by = Column(String(100), default="System Administrator")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class User(Base):
    __tablename__ = "users"

    id = Column(String(100), primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), default="")
    role = Column(String(50), default="operator")  # admin, operator, viewer, inspector, hub_manager, logistics, analytics_viewer
    assigned_hub_id = Column(String(100), ForeignKey("locations.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    avatar_url = Column(String(500), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_user_username", "username"),
        Index("idx_user_email", "email"),
    )

