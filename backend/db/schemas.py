from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict

# Product Schemas
class ProductBase(BaseModel):
    sku: str
    name: str
    category: str
    brand: Optional[str] = "HIEN Verified"
    variant: Optional[str] = "Standard Edition"
    unit_price: float
    selling_price: Optional[float] = None
    original_price: Optional[float] = None
    weight_kg: Optional[float] = 1.0
    image_url: Optional[str] = ""
    description: Optional[str] = ""
    active: Optional[bool] = True
    hygiene_sensitive: Optional[int] = 0
    fragile: Optional[int] = 0
    inspection_requirements: Optional[List[str]] = None
    return_eligibility_policy: Optional[str] = None

class ProductCreate(ProductBase):
    pass

class ProductOut(ProductBase):
    id: str
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

# Location Schemas
class LocationOut(BaseModel):
    id: str
    location_name: str
    city: str
    state: str
    latitude: float
    longitude: float
    storage_capacity_units: int
    current_inventory_units: int
    available_storage_units: int
    operating_hours: str
    delivery_radius_km: float
    model_config = ConfigDict(from_attributes=True)

# Inventory Bucket Schemas
class InventoryBucketOut(BaseModel):
    sku: str
    product_name: str
    category: str
    hub_id: str
    hub_name: str
    available_sellable: int
    reserved: int
    pending_qa: int
    refurbishment: int
    quarantined: int
    damaged: int
    in_transit: int
    total_stock: int
    storage_location: Optional[str] = "Bay A"
    last_updated: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

# Inventory Transaction Schema
class InventoryTransactionOut(BaseModel):
    id: str
    timestamp: datetime
    sku: str
    hub_id: str
    quantity_delta: int
    from_state: str
    to_state: str
    reason: str
    reference_id: str
    actor: str
    model_config = ConfigDict(from_attributes=True)

# Audit Ledger Schema
class AuditEventOut(BaseModel):
    id: str
    timestamp: datetime
    actor: str
    role: str
    action: str
    entity_type: str
    entity_id: str
    previous_state: Optional[str] = ""
    new_state: Optional[str] = ""
    result: str
    model_config = ConfigDict(from_attributes=True)

# Return Lifecycle Schemas
class ReturnRegisterReq(BaseModel):
    sku: str
    hub_id: str
    hub_name: Optional[str] = ""
    quantity: Optional[int] = 1
    product_name: Optional[str] = ""
    category: Optional[str] = "General"
    order_reference: Optional[str] = ""
    return_reason: Optional[str] = ""
    reported_condition: Optional[str] = "like_new"
    actor: Optional[str] = "Intake Associate"

class InspectionCheckReq(BaseModel):
    id: str
    label: str
    passed: Optional[bool] = None
    isMandatory: Optional[bool] = True
    notes: Optional[str] = ""

class PhysicalInspectionReq(BaseModel):
    disposition: str # approved_resale, refurbishment_required, quarantined, not_suitable
    inspector_name: Optional[str] = "Marcus Vance"
    inspector_notes: Optional[str] = ""
    checks: Optional[List[Dict[str, Any]]] = None

# Order Action Schemas
class OrderStockActionReq(BaseModel):
    order_id: str
    hub_id: str
    sku: str
    quantity: Optional[int] = 1
    actor: Optional[str] = "Order Fulfillment Agent"

# Transfer Action Schemas
class TransferActionReq(BaseModel):
    transfer_id: str
    source_hub_id: Optional[str] = "Tacoma Regional Logistics Hub"
    dest_hub_id: Optional[str] = "Seattle Central QA Hub"
    sku: str
    quantity: Optional[int] = 5
    actor: Optional[str] = "Logistics Coordinator"

# User & Authentication Schemas
class UserBase(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = ""
    role: Optional[str] = "operator"
    assigned_hub_id: Optional[str] = None
    avatar_url: Optional[str] = ""
    is_active: Optional[bool] = True

class UserRegisterReq(BaseModel):
    full_name: str
    email: str
    password: str
    confirm_password: Optional[str] = None
    role: Optional[str] = "operator"
    assigned_hub_id: Optional[str] = "LOC001"

class UserLoginReq(BaseModel):
    username_or_email: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    password: str
    role: Optional[str] = None

class UserOut(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str] = ""
    role: str
    assigned_hub_id: Optional[str] = None
    avatar_url: Optional[str] = ""
    is_active: bool
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

