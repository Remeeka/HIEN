import os
import json
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Set

class InventoryLedger:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.lock = threading.RLock()
        self.state_file = os.path.join(data_dir, 'inventory_realtime_state.json')
        self.transactions_file = os.path.join(data_dir, 'inventory_transactions.json')
        self.audit_file = os.path.join(data_dir, 'audit_ledger.json')
        
        # In-memory data structures
        self.inventory: Dict[str, Dict[str, Any]] = {} # key: f"{hub_id}:{sku}"
        self.transactions: List[Dict[str, Any]] = []
        self.audit_logs: List[Dict[str, Any]] = []
        self.returns_store: Dict[str, Dict[str, Any]] = {}
        self.orders_store: Dict[str, Dict[str, Any]] = {}
        self.transfers_store: Dict[str, Dict[str, Any]] = {}
        
        # Active SSE client subscriber queues
        self.subscribers: Set[asyncio.Queue] = set()
        
        self._load_or_initialize()

    def _get_key(self, hub_id: str, sku: str) -> str:
        return f"{hub_id.strip().lower()}:{sku.strip().upper()}"

    def _load_or_initialize(self):
        with self.lock:
            # Seed standard hubs and products
            hubs_seed = [
                {"id": "hub-sea-01", "name": "Seattle Central QA Hub"},
                {"id": "hub-bel-02", "name": "Bellevue Tech Exchange Store"},
                {"id": "hub-red-03", "name": "Redmond Micro-Fulfillment Center"},
                {"id": "hub-tac-04", "name": "Tacoma Regional Logistics Hub"},
                {"id": "hub-eve-05", "name": "Everett North Interchange Store"},
                {"id": "LOC001", "name": "Seattle Central QA Hub"},
                {"id": "LOC002", "name": "Bellevue Tech Exchange Store"},
                {"id": "LOC003", "name": "Redmond Micro-Fulfillment Center"},
                {"id": "LOC004", "name": "Tacoma Regional Logistics Hub"},
            ]
            products_seed = [
                {"sku": "SKU-AUD-801", "name": "AcoustiPro ANC Wireless Headphones", "category": "Audio & Sound", "unit_price": 199.99},
                {"sku": "SKU-CMP-204", "name": "SwiftBook Air Ultralight 14\"", "category": "Computing", "unit_price": 849.00},
                {"sku": "SKU-WR-405", "name": "PulseTrack Pro Smart Fitness Watch", "category": "Wearables", "unit_price": 169.50},
                {"sku": "SKU-CAM-902", "name": "LumixView 4K Creator Mirrorless Body", "category": "Consumer Electronics", "unit_price": 620.00},
                {"sku": "SKU-APP-703", "name": "BaristaTouch Precision Espresso Machine", "category": "Kitchen Appliances", "unit_price": 499.00},
                {"sku": "SKU-IOT-512", "name": "AeroMesh Wi-Fi 7 Tri-Band Node", "category": "Smart Home", "unit_price": 149.99},
            ]

            # Try to load existing persisted state
            if os.path.exists(self.state_file):
                try:
                    with open(self.state_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.inventory = data.get('inventory', {})
                        self.returns_store = data.get('returns', {})
                        self.orders_store = data.get('orders', {})
                        self.transfers_store = data.get('transfers', {})
                except Exception as e:
                    print(f"Warning: Could not read inventory state file: {e}")

            # If inventory empty, seed defaults
            if not self.inventory:
                for h in hubs_seed:
                    for p in products_seed:
                        key = self._get_key(h['id'], p['sku'])
                        
                        # Specific target seeds
                        if p['sku'] == 'SKU-AUD-801' and ('sea' in h['id'].lower() or h['id'] == 'LOC001'):
                            avail, res, pend, ref, quar, transit = 8, 2, 0, 0, 0, 0
                        elif p['sku'] == 'SKU-AUD-801' and ('tac' in h['id'].lower() or h['id'] == 'LOC004'):
                            avail, res, pend, ref, quar, transit = 14, 5, 0, 0, 0, 0
                        else:
                            avail, res, pend, ref, quar, transit = 6, 1, 0, 0, 0, 0
                            
                        self.inventory[key] = {
                            "sku": p['sku'],
                            "product_name": p['name'],
                            "category": p['category'],
                            "hub_id": h['id'],
                            "hub_name": h['name'],
                            "available_sellable": avail,
                            "reserved": res,
                            "pending_qa": pend,
                            "refurbishment": ref,
                            "quarantined": quar,
                            "damaged": 0,
                            "in_transit": transit,
                            "total_stock": avail + res + pend + ref + quar + transit,
                            "unit_price": p['unit_price'],
                            "last_updated": datetime.now().isoformat()
                        }
                self._save_state()

            if os.path.exists(self.transactions_file):
                try:
                    with open(self.transactions_file, 'r', encoding='utf-8') as f:
                        self.transactions = json.load(f)
                except Exception:
                    self.transactions = []

            if os.path.exists(self.audit_file):
                try:
                    with open(self.audit_file, 'r', encoding='utf-8') as f:
                        self.audit_logs = json.load(f)
                except Exception:
                    self.audit_logs = []

    def _save_state(self):
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "inventory": self.inventory,
                    "returns": self.returns_store,
                    "orders": self.orders_store,
                    "transfers": self.transfers_store
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save inventory state: {e}")

    def _save_transactions(self):
        try:
            with open(self.transactions_file, 'w', encoding='utf-8') as f:
                json.dump(self.transactions[-500:], f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save transactions: {e}")

    def _save_audit(self):
        try:
            with open(self.audit_file, 'w', encoding='utf-8') as f:
                json.dump(self.audit_logs[-500:], f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save audit logs: {e}")

    async def broadcast_event(self, event_type: str, payload: Dict[str, Any]):
        """Publish event to all active SSE subscribers."""
        event_data = {
            "type": event_type,
            "data": payload,
            "timestamp": datetime.now().isoformat()
        }
        dead_queues = set()
        for q in self.subscribers:
            try:
                q.put_nowait(event_data)
            except Exception:
                dead_queues.add(q)
        for dq in dead_queues:
            self.subscribers.discard(dq)

    def subscribe_events(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.subscribers.add(q)
        return q

    def unsubscribe_events(self, q: asyncio.Queue):
        self.subscribers.discard(q)

    def log_audit(self, actor: str, role: str, action: str, entity_type: str, entity_id: str,
                  previous_state: str = "", new_state: str = "", result: str = "SUCCESS", **kwargs):
        with self.lock:
            state_val = new_state or kwargs.get('newState', '')
            prev_val = previous_state or kwargs.get('previousState', '')
            event = {
                "id": f"aud-{int(datetime.now().timestamp() * 1000)}",
                "timestamp": datetime.now().isoformat(),
                "actor": actor,
                "role": role,
                "action": action,
                "entityType": entity_type,
                "entityId": entity_id,
                "previousState": prev_val,
                "newState": state_val,
                "result": result
            }
            self.audit_logs.insert(0, event)
            self._save_audit()
            return event

    def get_stock(self, hub_id: str, sku: str) -> Dict[str, Any]:
        with self.lock:
            key = self._get_key(hub_id, sku)
            if key not in self.inventory:
                # Try generic matching
                for k, v in self.inventory.items():
                    if v['sku'].upper() == sku.upper() and (hub_id.lower() in k or k.startswith(hub_id.lower())):
                        return v
                # Return empty default object
                return {
                    "sku": sku,
                    "hub_id": hub_id,
                    "available_sellable": 0,
                    "reserved": 0,
                    "pending_qa": 0,
                    "refurbishment": 0,
                    "quarantined": 0,
                    "damaged": 0,
                    "in_transit": 0,
                    "total_stock": 0,
                    "last_updated": datetime.now().isoformat()
                }
            return dict(self.inventory[key])

    def get_all_inventory(self, hub_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.lock:
            res = []
            for k, v in self.inventory.items():
                if hub_id and hub_id != 'all' and hub_id.lower() not in v['hub_id'].lower():
                    continue
                item = dict(v)
                item['total_stock'] = (
                    item.get('available_sellable', 0) +
                    item.get('reserved', 0) +
                    item.get('pending_qa', 0) +
                    item.get('refurbishment', 0) +
                    item.get('quarantined', 0) +
                    item.get('in_transit', 0)
                )
                res.append(item)
            return res

    def register_return(self, rma_number: str, sku: str, hub_id: str, hub_name: str,
                        quantity: int = 1, product_name: str = "", category: str = "General",
                        order_reference: str = "", actor: str = "Intake Associate") -> Dict[str, Any]:
        """
        Step 1: Return Registered
        Atomically increments pending_qa += quantity.
        Available sellable stock DOES NOT increase.
        """
        with self.lock:
            key = self._get_key(hub_id, sku)
            if key not in self.inventory:
                self.inventory[key] = {
                    "sku": sku,
                    "product_name": product_name or f"Product {sku}",
                    "category": category,
                    "hub_id": hub_id,
                    "hub_name": hub_name,
                    "available_sellable": 0,
                    "reserved": 0,
                    "pending_qa": 0,
                    "refurbishment": 0,
                    "quarantined": 0,
                    "damaged": 0,
                    "in_transit": 0,
                    "total_stock": 0,
                    "last_updated": datetime.now().isoformat()
                }

            inv = self.inventory[key]
            inv['pending_qa'] += quantity
            inv['total_stock'] = (
                inv['available_sellable'] + inv['reserved'] + inv['pending_qa'] +
                inv['refurbishment'] + inv['quarantined'] + inv['in_transit']
            )
            inv['last_updated'] = datetime.now().isoformat()

            # Record Return
            ret_record = {
                "id": rma_number,
                "rmaNumber": rma_number,
                "sku": sku,
                "productName": product_name or inv.get('product_name', sku),
                "category": category,
                "quantity": quantity,
                "hubId": hub_id,
                "hubName": hub_name,
                "orderReference": order_reference,
                "status": "registered",
                "inventory_restocked_at": None,
                "createdAt": datetime.now().isoformat(),
                "updatedAt": datetime.now().isoformat()
            }
            self.returns_store[rma_number] = ret_record

            # Record Transaction
            tx = {
                "id": f"tx-{int(datetime.now().timestamp() * 1000)}",
                "sku": sku,
                "hub_id": hub_id,
                "quantity_delta": quantity,
                "from_state": "CUSTOMER",
                "to_state": "PENDING_QA",
                "reason": "RETURN_REGISTERED",
                "reference_id": rma_number,
                "actor": actor,
                "timestamp": datetime.now().isoformat()
            }
            self.transactions.insert(0, tx)

            # Audit
            self.log_audit(
                actor=actor,
                role="Intake Associate",
                action="RETURN_REGISTERED",
                entity_type="ReturnItem",
                entity_id=rma_number,
                previous_state="Unregistered",
                newState=f"RMA {rma_number} registered. Pending QA: {inv['pending_qa']}, Available: {inv['available_sellable']}",
                result="SUCCESS"
            )

            self._save_state()
            self._save_transactions()

            return {
                "return": ret_record,
                "inventory": inv,
                "transaction": tx
            }

    def complete_physical_inspection(self, rma_number: str, disposition: str,
                                      inspector_name: str = "Marcus Vance",
                                      inspector_notes: str = "", checks: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Step 3: Physical QA Approval
        One atomic database transaction:
        - Verify return exists & not already restocked.
        - If approved_resale: pending_qa -= qty, available_sellable += qty.
        - If refurbishment_required: pending_qa -= qty, refurbishment += qty.
        - If quarantined: pending_qa -= qty, quarantined += qty.
        - If not_suitable: pending_qa -= qty, damaged += qty.
        """
        with self.lock:
            ret = self.returns_store.get(rma_number)
            if not ret:
                # Search by case-insensitive key
                for k, v in self.returns_store.items():
                    if k.lower() == rma_number.lower() or v.get('rmaNumber', '').lower() == rma_number.lower():
                        ret = v
                        break

            if not ret:
                raise ValueError(f"Return record '{rma_number}' not found.")

            if ret.get('inventory_restocked_at') is not None:
                raise ValueError(f"Return {rma_number} has already been processed into inventory at {ret.get('inventory_restocked_at')}. Duplicate restock blocked.")

            sku = ret['sku']
            hub_id = ret['hubId']
            qty = ret.get('quantity', 1)
            key = self._get_key(hub_id, sku)
            
            if key not in self.inventory:
                self.inventory[key] = {
                    "sku": sku,
                    "product_name": ret.get('productName', sku),
                    "category": ret.get('category', 'General'),
                    "hub_id": hub_id,
                    "hub_name": ret.get('hubName', hub_id),
                    "available_sellable": 0,
                    "reserved": 0,
                    "pending_qa": qty,
                    "refurbishment": 0,
                    "quarantined": 0,
                    "damaged": 0,
                    "in_transit": 0,
                    "total_stock": qty,
                    "last_updated": datetime.now().isoformat()
                }

            inv = self.inventory[key]
            
            # Decrement Pending QA (bounded at 0)
            inv['pending_qa'] = max(0, inv['pending_qa'] - qty)
            now_iso = datetime.now().isoformat()

            tx_reason = "RETURN_RESTOCK"
            to_state = "AVAILABLE_SELLABLE"

            if disposition == 'approved_resale':
                inv['available_sellable'] += qty
                to_state = "AVAILABLE_SELLABLE"
                tx_reason = "RETURN_RESTOCK"
                ret['status'] = "approved_resale"
            elif disposition == 'refurbishment_required':
                inv['refurbishment'] += qty
                to_state = "REFURBISHMENT"
                tx_reason = "REFURBISHMENT_ASSIGNED"
                ret['status'] = "refurbishment_required"
            elif disposition == 'quarantined':
                inv['quarantined'] += qty
                to_state = "QUARANTINED"
                tx_reason = "QUARANTINE_ENFORCED"
                ret['status'] = "quarantined"
            else: # not_suitable / salvage
                inv['damaged'] += qty
                to_state = "DAMAGED"
                tx_reason = "SALVAGE_ASSIGNED"
                ret['status'] = "not_suitable"

            inv['total_stock'] = (
                inv['available_sellable'] + inv['reserved'] + inv['pending_qa'] +
                inv['refurbishment'] + inv['quarantined'] + inv['in_transit']
            )
            inv['last_updated'] = now_iso

            ret['inventory_restocked_at'] = now_iso
            ret['updatedAt'] = now_iso
            ret['inspection'] = {
                "id": f"insp-{int(datetime.now().timestamp() * 1000)}",
                "inspectorName": inspector_name,
                "finalDisposition": disposition,
                "inspectorNotes": inspector_notes,
                "completedAt": now_iso,
                "checks": checks or []
            }

            # Create Transaction
            tx = {
                "id": f"tx-{int(datetime.now().timestamp() * 1000)}",
                "sku": sku,
                "hub_id": hub_id,
                "quantity_delta": qty,
                "from_state": "PENDING_QA",
                "to_state": to_state,
                "reason": tx_reason,
                "reference_id": rma_number,
                "actor": inspector_name,
                "timestamp": now_iso
            }
            self.transactions.insert(0, tx)

            # Audit Events
            self.log_audit(
                actor=inspector_name,
                role="Inspector",
                action="PHYSICAL_INSPECTION_APPROVED" if disposition == 'approved_resale' else "PHYSICAL_INSPECTION_COMPLETED",
                entity_type="PhysicalInspection",
                entity_id=rma_number,
                previous_state="assessment_pending",
                newState=f"Disposition: {disposition.upper()}",
                result="SUCCESS"
            )

            if disposition == 'approved_resale':
                self.log_audit(
                    actor=inspector_name,
                    role="Inspector",
                    action="INVENTORY_INCREASE_FROM_RETURN",
                    entity_type="InventoryItem",
                    entity_id=key,
                    previous_state=f"Available: {inv['available_sellable'] - qty}",
                    newState=f"Available: {inv['available_sellable']} (+{qty} from {rma_number})",
                    result="SUCCESS"
                )

            self._save_state()
            self._save_transactions()

            return {
                "return": ret,
                "inventory": inv,
                "transaction": tx
            }

    def reserve_order_stock(self, order_id: str, hub_id: str, sku: str, quantity: int = 1,
                            actor: str = "Order Fulfillment Agent") -> Dict[str, Any]:
        with self.lock:
            key = self._get_key(hub_id, sku)
            if key not in self.inventory or self.inventory[key]['available_sellable'] < quantity:
                avail = self.inventory.get(key, {}).get('available_sellable', 0)
                raise ValueError(f"Insufficient available sellable stock for {sku} at {hub_id}. Available: {avail}, Requested: {quantity}")

            inv = self.inventory[key]
            inv['available_sellable'] -= quantity
            inv['reserved'] += quantity
            inv['last_updated'] = datetime.now().isoformat()

            now_iso = datetime.now().isoformat()
            tx = {
                "id": f"tx-{int(datetime.now().timestamp() * 1000)}",
                "sku": sku,
                "hub_id": hub_id,
                "quantity_delta": -quantity,
                "from_state": "AVAILABLE_SELLABLE",
                "to_state": "RESERVED",
                "reason": "ORDER_RESERVED",
                "reference_id": order_id,
                "actor": actor,
                "timestamp": now_iso
            }
            self.transactions.insert(0, tx)

            self.log_audit(
                actor=actor,
                role="Hub Operator",
                action="ORDER_RESERVED",
                entity_type="LocalOrder",
                entity_id=order_id,
                previous_state=f"Available: {inv['available_sellable'] + quantity}",
                newState=f"Available: {inv['available_sellable']}, Reserved: {inv['reserved']}",
                result="SUCCESS"
            )

            self._save_state()
            self._save_transactions()
            return {"inventory": inv, "transaction": tx}

    def fulfill_order_stock(self, order_id: str, hub_id: str, sku: str, quantity: int = 1,
                            actor: str = "Courier Delivery Agent") -> Dict[str, Any]:
        with self.lock:
            key = self._get_key(hub_id, sku)
            if key not in self.inventory:
                raise ValueError(f"Inventory record {key} not found.")

            inv = self.inventory[key]
            inv['reserved'] = max(0, inv['reserved'] - quantity)
            inv['total_stock'] = (
                inv['available_sellable'] + inv['reserved'] + inv['pending_qa'] +
                inv['refurbishment'] + inv['quarantined'] + inv['in_transit']
            )
            inv['last_updated'] = datetime.now().isoformat()

            now_iso = datetime.now().isoformat()
            tx = {
                "id": f"tx-{int(datetime.now().timestamp() * 1000)}",
                "sku": sku,
                "hub_id": hub_id,
                "quantity_delta": -quantity,
                "from_state": "RESERVED",
                "to_state": "FULFILLED_CUSTOMER",
                "reason": "ORDER_FULFILLED",
                "reference_id": order_id,
                "actor": actor,
                "timestamp": now_iso
            }
            self.transactions.insert(0, tx)

            self.log_audit(
                actor=actor,
                role="Logistics",
                action="ORDER_FULFILLED",
                entity_type="LocalOrder",
                entity_id=order_id,
                previous_state=f"Reserved: {inv['reserved'] + quantity}",
                newState=f"Reserved: {inv['reserved']} (Total Stock: {inv['total_stock']})",
                result="SUCCESS"
            )

            self._save_state()
            self._save_transactions()
            return {"inventory": inv, "transaction": tx}

    def transfer_dispatch(self, transfer_id: str, source_hub_id: str, dest_hub_id: str,
                          sku: str, quantity: int = 5, actor: str = "David Kim") -> Dict[str, Any]:
        with self.lock:
            src_key = self._get_key(source_hub_id, sku)
            dest_key = self._get_key(dest_hub_id, sku)

            if src_key not in self.inventory:
                raise ValueError(f"Source inventory record {src_key} not found.")

            src_inv = self.inventory[src_key]
            # Deduct from reserved (or available if directly dispatched)
            if src_inv['reserved'] >= quantity:
                src_inv['reserved'] -= quantity
            else:
                src_inv['available_sellable'] = max(0, src_inv['available_sellable'] - quantity)

            src_inv['total_stock'] = (
                src_inv['available_sellable'] + src_inv['reserved'] + src_inv['pending_qa'] +
                src_inv['refurbishment'] + src_inv['quarantined'] + src_inv['in_transit']
            )
            src_inv['last_updated'] = datetime.now().isoformat()

            if dest_key not in self.inventory:
                self.inventory[dest_key] = {
                    "sku": sku,
                    "product_name": src_inv.get('product_name', sku),
                    "category": src_inv.get('category', 'General'),
                    "hub_id": dest_hub_id,
                    "hub_name": dest_hub_id,
                    "available_sellable": 0,
                    "reserved": 0,
                    "pending_qa": 0,
                    "refurbishment": 0,
                    "quarantined": 0,
                    "damaged": 0,
                    "in_transit": 0,
                    "total_stock": 0,
                    "last_updated": datetime.now().isoformat()
                }

            dest_inv = self.inventory[dest_key]
            dest_inv['in_transit'] += quantity
            dest_inv['total_stock'] = (
                dest_inv['available_sellable'] + dest_inv['reserved'] + dest_inv['pending_qa'] +
                dest_inv['refurbishment'] + dest_inv['quarantined'] + dest_inv['in_transit']
            )
            dest_inv['last_updated'] = datetime.now().isoformat()

            now_iso = datetime.now().isoformat()
            tx = {
                "id": f"tx-{int(datetime.now().timestamp() * 1000)}",
                "sku": sku,
                "hub_id": source_hub_id,
                "quantity_delta": -quantity,
                "from_state": "SOURCE_HUB",
                "to_state": "IN_TRANSIT",
                "reason": "TRANSFER_DISPATCHED",
                "reference_id": transfer_id,
                "actor": actor,
                "timestamp": now_iso
            }
            self.transactions.insert(0, tx)

            self.log_audit(
                actor=actor,
                role="Logistics",
                action="TRANSFER_DISPATCHED",
                entity_type="Transfer",
                entity_id=transfer_id,
                previous_state=f"Departed {source_hub_id}",
                newState=f"In transit to {dest_hub_id} ({quantity} units)",
                result="SUCCESS"
            )

            self._save_state()
            self._save_transactions()
            return {"source_inventory": src_inv, "dest_inventory": dest_inv, "transaction": tx}

    def transfer_receive(self, transfer_id: str, dest_hub_id: str, sku: str,
                         quantity: int = 5, actor: str = "Marcus Vance") -> Dict[str, Any]:
        with self.lock:
            dest_key = self._get_key(dest_hub_id, sku)
            if dest_key not in self.inventory:
                raise ValueError(f"Destination inventory record {dest_key} not found.")

            dest_inv = self.inventory[dest_key]
            dest_inv['in_transit'] = max(0, dest_inv['in_transit'] - quantity)
            dest_inv['available_sellable'] += quantity
            dest_inv['total_stock'] = (
                dest_inv['available_sellable'] + dest_inv['reserved'] + dest_inv['pending_qa'] +
                dest_inv['refurbishment'] + dest_inv['quarantined'] + dest_inv['in_transit']
            )
            dest_inv['last_updated'] = datetime.now().isoformat()

            now_iso = datetime.now().isoformat()
            tx = {
                "id": f"tx-{int(datetime.now().timestamp() * 1000)}",
                "sku": sku,
                "hub_id": dest_hub_id,
                "quantity_delta": quantity,
                "from_state": "IN_TRANSIT",
                "to_state": "AVAILABLE_SELLABLE",
                "reason": "TRANSFER_RECEIVED",
                "reference_id": transfer_id,
                "actor": actor,
                "timestamp": now_iso
            }
            self.transactions.insert(0, tx)

            self.log_audit(
                actor=actor,
                role="QA Inspector",
                action="TRANSFER_RECEIVED",
                entity_type="Transfer",
                entity_id=transfer_id,
                previous_state="Inbound Transit",
                newState=f"Received at {dest_hub_id}. Available: {dest_inv['available_sellable']} (+{quantity})",
                result="SUCCESS"
            )

            self._save_state()
            self._save_transactions()
            return {"dest_inventory": dest_inv, "transaction": tx}
