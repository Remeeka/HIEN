import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, func
from backend.db import models

def log_audit_in_db(
    db: Session,
    actor: str,
    role: str,
    action: str,
    entity_type: str,
    entity_id: str,
    previous_state: str = "",
    new_state: str = "",
    result: str = "SUCCESS",
    metadata_json: Optional[Dict[str, Any]] = None
) -> models.AuditLedger:
    audit_id = f"aud-{int(datetime.utcnow().timestamp() * 1000)}-{uuid.uuid4().hex[:6]}"
    entry = models.AuditLedger(
        id=audit_id,
        timestamp=datetime.utcnow(),
        actor=actor,
        role=role,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        previous_state=previous_state,
        new_state=new_state,
        result=result,
        metadata_json=metadata_json or {}
    )
    db.add(entry)
    return entry


def get_or_create_inventory(
    db: Session,
    sku: str,
    hub_id: str,
    hub_name: str = "",
    product_name: str = "",
    category: str = "General",
    for_update: bool = False
) -> models.Inventory:
    query = db.query(models.Inventory).filter(
        models.Inventory.sku == sku,
        models.Inventory.hub_id == hub_id
    )
    if for_update and "sqlite" not in str(db.bind.url):
        query = query.with_for_update()
    
    inv = query.first()
    if not inv:
        inv = models.Inventory(
            id=f"inv-{hub_id}-{sku}".lower(),
            sku=sku,
            product_name=product_name or f"Product {sku}",
            category=category,
            variant="Standard",
            hub_id=hub_id,
            hub_name=hub_name or hub_id,
            available_sellable=0,
            reserved=0,
            pending_qa=0,
            refurbishment=0,
            quarantined=0,
            damaged=0,
            in_transit=0,
            total_stock=0,
            last_updated=datetime.utcnow()
        )
        db.add(inv)
        db.flush()
    return inv


def register_return_in_db(
    db: Session,
    rma_number: str,
    sku: str,
    hub_id: str,
    hub_name: str,
    quantity: int = 1,
    product_name: str = "",
    category: str = "General",
    order_reference: str = "",
    return_reason: str = "",
    reported_condition: str = "like_new",
    actor: str = "Intake Associate"
) -> Dict[str, Any]:
    """
    Step 1: Return Registration Transaction
    - Atomically increments inventory.pending_qa += quantity
    - Available sellable stock DOES NOT increase (+0)
    - Records immutable RETURN_REGISTERED inventory transaction
    - Records RETURN_REGISTERED audit log
    """
    inv = get_or_create_inventory(db, sku, hub_id, hub_name, product_name, category, for_update=True)
    
    # Increment Pending QA only
    inv.pending_qa += quantity
    inv.total_stock = (
        inv.available_sellable + inv.reserved + inv.pending_qa +
        inv.refurbishment + inv.quarantined + inv.damaged + inv.in_transit
    )
    inv.last_updated = datetime.utcnow()

    # Create Return Record
    ret = models.Return(
        id=rma_number,
        rma_number=rma_number,
        order_reference=order_reference or f"ORD-REF-{int(datetime.utcnow().timestamp())}",
        sku=sku,
        product_name=product_name or inv.product_name,
        category=category,
        quantity=quantity,
        hub_id=hub_id,
        hub_name=hub_name or inv.hub_name,
        return_reason=return_reason,
        reported_condition=reported_condition,
        status="registered",
        inventory_restocked_at=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(ret)

    # Record Transaction
    tx = models.InventoryTransaction(
        id=f"tx-{int(datetime.utcnow().timestamp() * 1000)}-{uuid.uuid4().hex[:6]}",
        timestamp=datetime.utcnow(),
        sku=sku,
        product_name=inv.product_name,
        hub_id=hub_id,
        quantity_delta=quantity,
        from_state="CUSTOMER",
        to_state="PENDING_QA",
        reason="RETURN_REGISTERED",
        reference_id=rma_number,
        actor=actor,
        idempotency_key=f"ret-reg-{rma_number}"
    )
    db.add(tx)

    # Audit log
    log_audit_in_db(
        db=db,
        actor=actor,
        role="Intake Associate",
        action="RETURN_REGISTERED",
        entity_type="ReturnItem",
        entity_id=rma_number,
        previous_state="Unregistered",
        new_state=f"RMA {rma_number} registered. Pending QA: {inv.pending_qa}, Available: {inv.available_sellable}",
        result="SUCCESS"
    )

    db.commit()
    db.refresh(inv)
    db.refresh(ret)

    return {
        "return": {
            "id": ret.id,
            "rmaNumber": ret.rma_number,
            "sku": ret.sku,
            "productName": ret.product_name,
            "category": ret.category,
            "quantity": ret.quantity,
            "hubId": ret.hub_id,
            "hubName": ret.hub_name,
            "status": ret.status,
            "orderReference": ret.order_reference,
            "createdAt": ret.created_at.isoformat() if ret.created_at else "",
            "updatedAt": ret.updated_at.isoformat() if ret.updated_at else ""
        },
        "inventory": {
            "sku": inv.sku,
            "hub_id": inv.hub_id,
            "available_sellable": inv.available_sellable,
            "reserved": inv.reserved,
            "pending_qa": inv.pending_qa,
            "refurbishment": inv.refurbishment,
            "quarantined": inv.quarantined,
            "damaged": inv.damaged,
            "in_transit": inv.in_transit,
            "total_stock": inv.total_stock,
            "last_updated": inv.last_updated.isoformat() if inv.last_updated else ""
        },
        "transaction": {
            "id": tx.id,
            "sku": tx.sku,
            "hub_id": tx.hub_id,
            "quantity_delta": tx.quantity_delta,
            "from_state": tx.from_state,
            "to_state": tx.to_state,
            "reason": tx.reason,
            "reference_id": tx.reference_id,
            "actor": tx.actor,
            "timestamp": tx.timestamp.isoformat() if tx.timestamp else ""
        }
    }


def complete_inspection_in_db(
    db: Session,
    rma_number: str,
    disposition: str,
    inspector_name: str = "Marcus Vance",
    inspector_notes: str = "",
    checks: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Step 3: Physical QA Approval Atomic Transaction
    - Validates return existence & enforces Idempotency Guard (no double-restock).
    - If approved_resale: pending_qa -= quantity, available_sellable += quantity
    - If refurbishment_required: pending_qa -= quantity, refurbishment += quantity
    - If quarantined: pending_qa -= quantity, quarantined += quantity
    - If not_suitable: pending_qa -= quantity, damaged += quantity
    - Emits atomic inventory transaction & audit events.
    """
    ret_query = db.query(models.Return).filter(
        or_(models.Return.id == rma_number, models.Return.rma_number == rma_number)
    )
    if "sqlite" not in str(db.bind.url):
        ret_query = ret_query.with_for_update()
    ret = ret_query.first()

    if not ret:
        raise ValueError(f"Return record '{rma_number}' not found.")

    if ret.inventory_restocked_at is not None:
        raise ValueError(f"Return {rma_number} has already been processed into inventory at {ret.inventory_restocked_at}. Duplicate restock blocked.")

    sku = ret.sku
    hub_id = ret.hub_id
    qty = ret.quantity or 1

    inv = get_or_create_inventory(db, sku, hub_id, ret.hub_name, ret.product_name, ret.category, for_update=True)
    
    # Decrement pending_qa
    inv.pending_qa = max(0, inv.pending_qa - qty)
    now_utc = datetime.utcnow()

    to_state = "AVAILABLE_SELLABLE"
    tx_reason = "RETURN_RESTOCK"

    if disposition == "approved_resale":
        inv.available_sellable += qty
        to_state = "AVAILABLE_SELLABLE"
        tx_reason = "RETURN_RESTOCK"
        ret.status = "approved_resale"
    elif disposition == "refurbishment_required":
        inv.refurbishment += qty
        to_state = "REFURBISHMENT"
        tx_reason = "REFURBISHMENT_ASSIGNED"
        ret.status = "refurbishment_required"
    elif disposition == "quarantined":
        inv.quarantined += qty
        to_state = "QUARANTINED"
        tx_reason = "QUARANTINE_ENFORCED"
        ret.status = "quarantined"
    else:
        inv.damaged += qty
        to_state = "DAMAGED"
        tx_reason = "SALVAGE_ASSIGNED"
        ret.status = "not_suitable"

    inv.total_stock = (
        inv.available_sellable + inv.reserved + inv.pending_qa +
        inv.refurbishment + inv.quarantined + inv.damaged + inv.in_transit
    )
    inv.last_updated = now_utc

    ret.inventory_restocked_at = now_utc
    ret.updated_at = now_utc

    # Create Physical Inspection record
    insp = models.PhysicalInspection(
        id=f"insp-{int(now_utc.timestamp() * 1000)}-{uuid.uuid4().hex[:6]}",
        return_id=ret.id,
        inspector_id="u-2",
        inspector_name=inspector_name,
        checks_json=checks or [],
        final_disposition=disposition,
        inspector_notes=inspector_notes,
        completed_at=now_utc
    )
    db.add(insp)

    # Create Transaction
    tx = models.InventoryTransaction(
        id=f"tx-{int(now_utc.timestamp() * 1000)}-{uuid.uuid4().hex[:6]}",
        timestamp=now_utc,
        sku=sku,
        product_name=inv.product_name,
        hub_id=hub_id,
        quantity_delta=qty,
        from_state="PENDING_QA",
        to_state=to_state,
        reason=tx_reason,
        reference_id=rma_number,
        actor=inspector_name,
        idempotency_key=f"insp-restock-{rma_number}"
    )
    db.add(tx)

    # Audit Events
    log_audit_in_db(
        db=db,
        actor=inspector_name,
        role="Inspector",
        action="PHYSICAL_INSPECTION_APPROVED" if disposition == "approved_resale" else "PHYSICAL_INSPECTION_COMPLETED",
        entity_type="PhysicalInspection",
        entity_id=rma_number,
        previous_state="assessment_pending",
        new_state=f"Disposition: {disposition.upper()}",
        result="SUCCESS"
    )

    if disposition == "approved_resale":
        log_audit_in_db(
            db=db,
            actor=inspector_name,
            role="Inspector",
            action="INVENTORY_INCREASE_FROM_RETURN",
            entity_type="InventoryItem",
            entity_id=f"{hub_id}:{sku}",
            previous_state=f"Available: {inv.available_sellable - qty}",
            new_state=f"Available: {inv.available_sellable} (+{qty} from {rma_number})",
            result="SUCCESS"
        )

    db.commit()
    db.refresh(inv)
    db.refresh(ret)

    return {
        "return": {
            "id": ret.id,
            "rmaNumber": ret.rma_number,
            "sku": ret.sku,
            "productName": ret.product_name,
            "status": ret.status,
            "hubId": ret.hub_id,
            "hubName": ret.hub_name,
            "inventory_restocked_at": ret.inventory_restocked_at.isoformat() if ret.inventory_restocked_at else None
        },
        "inventory": {
            "sku": inv.sku,
            "hub_id": inv.hub_id,
            "available_sellable": inv.available_sellable,
            "reserved": inv.reserved,
            "pending_qa": inv.pending_qa,
            "refurbishment": inv.refurbishment,
            "quarantined": inv.quarantined,
            "damaged": inv.damaged,
            "in_transit": inv.in_transit,
            "total_stock": inv.total_stock,
            "last_updated": inv.last_updated.isoformat() if inv.last_updated else ""
        },
        "transaction": {
            "id": tx.id,
            "sku": tx.sku,
            "hub_id": tx.hub_id,
            "quantity_delta": tx.quantity_delta,
            "from_state": tx.from_state,
            "to_state": tx.to_state,
            "reason": tx.reason,
            "reference_id": tx.reference_id,
            "actor": tx.actor,
            "timestamp": tx.timestamp.isoformat() if tx.timestamp else ""
        }
    }


def reserve_order_stock_in_db(
    db: Session,
    order_id: str,
    hub_id: str,
    sku: str,
    quantity: int = 1,
    actor: str = "Order Fulfillment Agent"
) -> Dict[str, Any]:
    """Zero Double Allocation stock reservation with row-level lock."""
    inv = get_or_create_inventory(db, sku, hub_id, for_update=True)
    if inv.available_sellable < quantity:
        raise ValueError(f"Insufficient available sellable stock for {sku} at {hub_id}. Available: {inv.available_sellable}, Requested: {quantity}")

    inv.available_sellable -= quantity
    inv.reserved += quantity
    inv.last_updated = datetime.utcnow()

    now_utc = datetime.utcnow()
    tx = models.InventoryTransaction(
        id=f"tx-{int(now_utc.timestamp() * 1000)}-{uuid.uuid4().hex[:6]}",
        timestamp=now_utc,
        sku=sku,
        product_name=inv.product_name,
        hub_id=hub_id,
        quantity_delta=-quantity,
        from_state="AVAILABLE_SELLABLE",
        to_state="RESERVED",
        reason="ORDER_RESERVED",
        reference_id=order_id,
        actor=actor,
        idempotency_key=f"ord-res-{order_id}-{int(now_utc.timestamp())}"
    )
    db.add(tx)

    log_audit_in_db(
        db=db,
        actor=actor,
        role="Hub Operator",
        action="ORDER_RESERVED",
        entity_type="LocalOrder",
        entity_id=order_id,
        previous_state=f"Available: {inv.available_sellable + quantity}",
        new_state=f"Available: {inv.available_sellable}, Reserved: {inv.reserved}",
        result="SUCCESS"
    )

    db.commit()
    db.refresh(inv)

    return {
        "inventory": {
            "sku": inv.sku,
            "hub_id": inv.hub_id,
            "available_sellable": inv.available_sellable,
            "reserved": inv.reserved,
            "pending_qa": inv.pending_qa,
            "refurbishment": inv.refurbishment,
            "quarantined": inv.quarantined,
            "damaged": inv.damaged,
            "in_transit": inv.in_transit,
            "total_stock": inv.total_stock
        },
        "transaction": {
            "id": tx.id,
            "sku": tx.sku,
            "hub_id": tx.hub_id,
            "quantity_delta": tx.quantity_delta,
            "reason": tx.reason,
            "reference_id": tx.reference_id
        }
    }


def fulfill_order_stock_in_db(
    db: Session,
    order_id: str,
    hub_id: str,
    sku: str,
    quantity: int = 1,
    actor: str = "Courier Delivery Agent"
) -> Dict[str, Any]:
    inv = get_or_create_inventory(db, sku, hub_id, for_update=True)
    inv.reserved = max(0, inv.reserved - quantity)
    inv.total_stock = (
        inv.available_sellable + inv.reserved + inv.pending_qa +
        inv.refurbishment + inv.quarantined + inv.damaged + inv.in_transit
    )
    inv.last_updated = datetime.utcnow()

    now_utc = datetime.utcnow()
    tx = models.InventoryTransaction(
        id=f"tx-{int(now_utc.timestamp() * 1000)}-{uuid.uuid4().hex[:6]}",
        timestamp=now_utc,
        sku=sku,
        product_name=inv.product_name,
        hub_id=hub_id,
        quantity_delta=-quantity,
        from_state="RESERVED",
        to_state="FULFILLED_CUSTOMER",
        reason="ORDER_FULFILLED",
        reference_id=order_id,
        actor=actor,
        idempotency_key=f"ord-ful-{order_id}-{int(now_utc.timestamp())}"
    )
    db.add(tx)

    log_audit_in_db(
        db=db,
        actor=actor,
        role="Logistics",
        action="ORDER_FULFILLED",
        entity_type="LocalOrder",
        entity_id=order_id,
        previous_state=f"Reserved: {inv.reserved + quantity}",
        new_state=f"Reserved: {inv.reserved} (Total: {inv.total_stock})",
        result="SUCCESS"
    )

    db.commit()
    db.refresh(inv)

    return {
        "inventory": {
            "sku": inv.sku,
            "hub_id": inv.hub_id,
            "available_sellable": inv.available_sellable,
            "reserved": inv.reserved,
            "total_stock": inv.total_stock
        },
        "transaction": {
            "id": tx.id,
            "sku": tx.sku,
            "hub_id": tx.hub_id,
            "quantity_delta": tx.quantity_delta,
            "reason": tx.reason
        }
    }


def dispatch_transfer_in_db(
    db: Session,
    transfer_id: str,
    source_hub_id: str,
    dest_hub_id: str,
    sku: str,
    quantity: int = 5,
    actor: str = "David Kim"
) -> Dict[str, Any]:
    src_inv = get_or_create_inventory(db, sku, source_hub_id, for_update=True)
    dest_inv = get_or_create_inventory(db, sku, dest_hub_id, for_update=True)

    if src_inv.reserved >= quantity:
        src_inv.reserved -= quantity
    else:
        src_inv.available_sellable = max(0, src_inv.available_sellable - quantity)

    src_inv.total_stock = (
        src_inv.available_sellable + src_inv.reserved + src_inv.pending_qa +
        src_inv.refurbishment + src_inv.quarantined + src_inv.damaged + src_inv.in_transit
    )
    src_inv.last_updated = datetime.utcnow()

    dest_inv.in_transit += quantity
    dest_inv.total_stock = (
        dest_inv.available_sellable + dest_inv.reserved + dest_inv.pending_qa +
        dest_inv.refurbishment + dest_inv.quarantined + dest_inv.damaged + dest_inv.in_transit
    )
    dest_inv.last_updated = datetime.utcnow()

    now_utc = datetime.utcnow()
    tx = models.InventoryTransaction(
        id=f"tx-{int(now_utc.timestamp() * 1000)}-{uuid.uuid4().hex[:6]}",
        timestamp=now_utc,
        sku=sku,
        product_name=src_inv.product_name,
        hub_id=source_hub_id,
        source_hub_id=source_hub_id,
        destination_hub_id=dest_hub_id,
        quantity_delta=-quantity,
        from_state="SOURCE_HUB",
        to_state="IN_TRANSIT",
        reason="TRANSFER_DISPATCHED",
        reference_id=transfer_id,
        actor=actor,
        idempotency_key=f"tr-disp-{transfer_id}-{int(now_utc.timestamp())}"
    )
    db.add(tx)

    log_audit_in_db(
        db=db,
        actor=actor,
        role="Logistics",
        action="TRANSFER_DISPATCHED",
        entity_type="Transfer",
        entity_id=transfer_id,
        previous_state=f"Departed {source_hub_id}",
        new_state=f"In transit to {dest_hub_id} ({quantity} units)",
        result="SUCCESS"
    )

    db.commit()
    db.refresh(src_inv)
    db.refresh(dest_inv)

    return {
        "source_inventory": {"sku": src_inv.sku, "hub_id": src_inv.hub_id, "available_sellable": src_inv.available_sellable, "total_stock": src_inv.total_stock},
        "dest_inventory": {"sku": dest_inv.sku, "hub_id": dest_inv.hub_id, "in_transit": dest_inv.in_transit, "total_stock": dest_inv.total_stock}
    }


def receive_transfer_in_db(
    db: Session,
    transfer_id: str,
    dest_hub_id: str,
    sku: str,
    quantity: int = 5,
    actor: str = "Marcus Vance"
) -> Dict[str, Any]:
    dest_inv = get_or_create_inventory(db, sku, dest_hub_id, for_update=True)
    dest_inv.in_transit = max(0, dest_inv.in_transit - quantity)
    dest_inv.available_sellable += quantity
    dest_inv.total_stock = (
        dest_inv.available_sellable + dest_inv.reserved + dest_inv.pending_qa +
        dest_inv.refurbishment + dest_inv.quarantined + dest_inv.damaged + dest_inv.in_transit
    )
    dest_inv.last_updated = datetime.utcnow()

    now_utc = datetime.utcnow()
    tx = models.InventoryTransaction(
        id=f"tx-{int(now_utc.timestamp() * 1000)}-{uuid.uuid4().hex[:6]}",
        timestamp=now_utc,
        sku=sku,
        product_name=dest_inv.product_name,
        hub_id=dest_hub_id,
        quantity_delta=quantity,
        from_state="IN_TRANSIT",
        to_state="AVAILABLE_SELLABLE",
        reason="TRANSFER_RECEIVED",
        reference_id=transfer_id,
        actor=actor,
        idempotency_key=f"tr-rec-{transfer_id}-{int(now_utc.timestamp())}"
    )
    db.add(tx)

    log_audit_in_db(
        db=db,
        actor=actor,
        role="QA Inspector",
        action="TRANSFER_RECEIVED",
        entity_type="Transfer",
        entity_id=transfer_id,
        previous_state="Inbound Transit",
        new_state=f"Received at {dest_hub_id}. Available: {dest_inv.available_sellable} (+{quantity})",
        result="SUCCESS"
    )

    db.commit()
    db.refresh(dest_inv)

    return {
        "dest_inventory": {
            "sku": dest_inv.sku,
            "hub_id": dest_inv.hub_id,
            "available_sellable": dest_inv.available_sellable,
            "in_transit": dest_inv.in_transit,
            "total_stock": dest_inv.total_stock
        }
    }


def get_realtime_inventory_from_db(db: Session, hub_id: Optional[str] = None) -> List[Dict[str, Any]]:
    query = db.query(models.Inventory)
    if hub_id and hub_id != "all":
        query = query.filter(models.Inventory.hub_id.ilike(f"%{hub_id}%"))
    
    items = query.all()
    res = []
    for inv in items:
        res.append({
            "sku": inv.sku,
            "product_name": inv.product_name,
            "category": inv.category,
            "variant": inv.variant,
            "hub_id": inv.hub_id,
            "hub_name": inv.hub_name,
            "available_sellable": inv.available_sellable,
            "reserved": inv.reserved,
            "pending_qa": inv.pending_qa,
            "refurbishment": inv.refurbishment,
            "quarantined": inv.quarantined,
            "damaged": inv.damaged,
            "in_transit": inv.in_transit,
            "total_stock": inv.total_stock,
            "storage_location": inv.storage_location,
            "unit_cost": inv.unit_cost,
            "selling_price": inv.selling_price,
            "last_updated": inv.last_updated.isoformat() if inv.last_updated else ""
        })
    return res


def get_inventory_transactions_from_db(db: Session, limit: int = 50) -> List[Dict[str, Any]]:
    items = db.query(models.InventoryTransaction).order_by(models.InventoryTransaction.timestamp.desc()).limit(limit).all()
    return [{
        "id": tx.id,
        "timestamp": tx.timestamp.isoformat() if tx.timestamp else "",
        "sku": tx.sku,
        "product_name": tx.product_name,
        "hub_id": tx.hub_id,
        "quantity_delta": tx.quantity_delta,
        "from_state": tx.from_state,
        "to_state": tx.to_state,
        "reason": tx.reason,
        "reference_id": tx.reference_id,
        "actor": tx.actor
    } for tx in items]


def get_audit_logs_from_db(db: Session, limit: int = 50) -> List[Dict[str, Any]]:
    items = db.query(models.AuditLedger).order_by(models.AuditLedger.timestamp.desc()).limit(limit).all()
    return [{
        "id": a.id,
        "timestamp": a.timestamp.isoformat() if a.timestamp else "",
        "actor": a.actor,
        "role": a.role,
        "action": a.action,
        "entityType": a.entity_type,
        "entityId": a.entity_id,
        "previousState": a.previous_state,
        "newState": a.new_state,
        "result": a.result
    } for a in items]


def create_product_in_db(db: Session, payload: Dict[str, Any]) -> models.Product:
    sku = payload["sku"].strip()
    existing = db.query(models.Product).filter(models.Product.sku.ilike(sku)).first()
    if existing:
        raise ValueError(f"Product with SKU '{sku}' already exists.")
    
    prod = models.Product(
        id=sku,
        sku=sku,
        name=payload["name"].strip(),
        category=payload["category"].strip(),
        brand=payload.get("brand", "HIEN Verified"),
        variant=payload.get("variant", "Standard Edition"),
        unit_price=float(payload["unit_price"]),
        selling_price=float(payload.get("selling_price") or payload["unit_price"]),
        original_price=float(payload.get("original_price") or round(float(payload["unit_price"]) * 1.35, 2)),
        weight_kg=float(payload.get("weight_kg", 1.0)),
        image_url=payload.get("image_url", ""),
        description=payload.get("description", ""),
        active=bool(payload.get("active", True)),
        inspection_requirements=payload.get("inspection_requirements", []),
        return_eligibility_policy=payload.get("return_eligibility_policy", "Standard 30-day return policy.")
    )
    db.add(prod)
    db.commit()
    db.refresh(prod)
    return prod
