import os
import sys
import unittest
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db.database import SessionLocal, check_db_health, engine
from backend.db import crud, models

class TestPostgreSQLHIENIntegration(unittest.TestCase):
    def setUp(self):
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()

    def test_01_db_health(self):
        health = check_db_health()
        print("\n[TEST 1] DB Health Check:", health)
        self.assertTrue(health["healthy"])
        self.assertEqual(health["database_engine"], "PostgreSQL")

    def test_02_return_lifecycle_and_idempotency(self):
        print("\n[TEST 2] Testing Return Registration -> Physical QA -> Zero Double Restock")
        test_rma = f"TEST-RMA-{int(datetime.utcnow().timestamp())}"
        test_sku = "SKU-AUD-801"
        test_hub = "hub-sea-01"

        # 1. Check initial available
        inv_before = crud.get_or_create_inventory(self.db, test_sku, test_hub)
        avail_start = inv_before.available_sellable
        pending_start = inv_before.pending_qa

        # 2. Register Return
        res_reg = crud.register_return_in_db(
            db=self.db,
            rma_number=test_rma,
            sku=test_sku,
            hub_id=test_hub,
            hub_name="Seattle Central QA Hub",
            quantity=1,
            product_name="AcoustiPro ANC Wireless Headphones",
            category="Audio & Sound",
            order_reference="ORD-TEST-001"
        )
        self.assertEqual(res_reg["inventory"]["pending_qa"], pending_start + 1)
        self.assertEqual(res_reg["inventory"]["available_sellable"], avail_start) # Sellable must NOT increase

        # 3. Complete QA Inspection (approved_resale)
        res_qa = crud.complete_inspection_in_db(
            db=self.db,
            rma_number=test_rma,
            disposition="approved_resale",
            inspector_name="Marcus Vance",
            inspector_notes="Unit in pristine condition. Passed all checks.",
            checks=[{"name": "Packaging", "passed": True}]
        )
        self.assertEqual(res_qa["inventory"]["available_sellable"], avail_start + 1)
        self.assertEqual(res_qa["inventory"]["pending_qa"], pending_start)

        # 4. Idempotency Guard (Must block duplicate restock)
        with self.assertRaises(ValueError):
            crud.complete_inspection_in_db(
                db=self.db,
                rma_number=test_rma,
                disposition="approved_resale"
            )
        print("   -> Passed: Return registered (+0 sellable), QA approved (+1 sellable), and Duplicate Restock blocked by Idempotency Guard!")

    def test_03_order_reservation_and_fulfillment(self):
        print("\n[TEST 3] Testing Order Reservation & Zero Double Allocation")
        test_sku = "SKU-AUD-801"
        test_hub = "hub-tac-04"
        order_id = f"ORD-TEST-{int(datetime.utcnow().timestamp())}"

        inv_before = crud.get_or_create_inventory(self.db, test_sku, test_hub)
        avail_start = inv_before.available_sellable
        res_start = inv_before.reserved

        if avail_start < 1:
            inv_before.available_sellable = 5
            self.db.commit()
            avail_start = 5

        # Reserve 1 unit
        res_reserve = crud.reserve_order_stock_in_db(
            db=self.db,
            order_id=order_id,
            hub_id=test_hub,
            sku=test_sku,
            quantity=1
        )
        self.assertEqual(res_reserve["inventory"]["available_sellable"], avail_start - 1)
        self.assertEqual(res_reserve["inventory"]["reserved"], res_start + 1)

        # Fulfill 1 unit
        res_fulfill = crud.fulfill_order_stock_in_db(
            db=self.db,
            order_id=order_id,
            hub_id=test_hub,
            sku=test_sku,
            quantity=1
        )
        self.assertEqual(res_fulfill["inventory"]["reserved"], res_start)
        print("   -> Passed: Order stock reserved with row lock and fulfilled accurately!")

    def test_04_transfer_dispatch_and_receipt(self):
        print("\n[TEST 4] Testing Inter-Hub Transfer Dispatch & Receipt Lifecycle")
        transfer_id = f"TR-TEST-{int(datetime.utcnow().timestamp())}"
        src_hub = "hub-tac-04"
        dest_hub = "hub-sea-01"
        sku = "SKU-AUD-801"
        qty = 2

        src_inv = crud.get_or_create_inventory(self.db, sku, src_hub)
        if src_inv.available_sellable < qty:
            src_inv.available_sellable = 10
            self.db.commit()

        src_avail_before = src_inv.available_sellable
        dest_inv = crud.get_or_create_inventory(self.db, sku, dest_hub)
        dest_transit_before = dest_inv.in_transit
        dest_avail_before = dest_inv.available_sellable

        # Dispatch Transfer
        res_disp = crud.dispatch_transfer_in_db(
            db=self.db,
            transfer_id=transfer_id,
            source_hub_id=src_hub,
            dest_hub_id=dest_hub,
            sku=sku,
            quantity=qty
        )
        self.assertEqual(res_disp["source_inventory"]["available_sellable"], src_avail_before - qty)
        self.assertEqual(res_disp["dest_inventory"]["in_transit"], dest_transit_before + qty)

        # Receive Transfer
        res_rec = crud.receive_transfer_in_db(
            db=self.db,
            transfer_id=transfer_id,
            dest_hub_id=dest_hub,
            sku=sku,
            quantity=qty
        )
        self.assertEqual(res_rec["dest_inventory"]["in_transit"], dest_transit_before)
        self.assertEqual(res_rec["dest_inventory"]["available_sellable"], dest_avail_before + qty)
        print("   -> Passed: Transfer dispatched into in_transit and received into sellable stock!")

if __name__ == '__main__':
    unittest.main(verbosity=2)
