#!/usr/bin/env python3
"""
Clean up mock encounter data before stakeholder demo.
Run this script from the hospital repository root.

Usage:
    python scripts/cleanup_mock_data.py [--dry-run]
"""

import sys
from app import app
from extensions import db
from departments.models.encounter import Encounter
from departments.models.billing import InvoiceLineItem


def cleanup_mock_data(dry_run=False):
    """Remove mock encounters and their associated billing records."""
    with app.app_context():
        # Find mock encounters
        mock_encounters = Encounter.query.filter(
            Encounter.chief_complaint.like("Mock %")
        ).all()
        
        count = len(mock_encounters)
        print(f"Found {count} mock encounters")
        
        if dry_run:
            print("DRY RUN — no changes made")
            for enc in mock_encounters[:10]:  # Show first 10
                print(f"  - Encounter #{enc.id}: {enc.encounter_type} - {enc.chief_complaint}")
            if count > 10:
                print(f"  ... and {count - 10} more")
            return count
        
        # Delete associated line items first (FK constraint)
        encounter_ids = [enc.id for enc in mock_encounters]
        if encounter_ids:
            line_items_deleted = InvoiceLineItem.query.filter(
                InvoiceLineItem.encounter_id.in_(encounter_ids)
            ).delete(synchronize_session=False)
            print(f"Deleted {line_items_deleted} associated line items")
        
        # Delete the mock encounters
        deleted = Encounter.query.filter(
            Encounter.chief_complaint.like("Mock %")
        ).delete(synchronize_session=False)
        
        db.session.commit()
        print(f"✅ Deleted {deleted} mock encounters")
        return deleted


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    cleanup_mock_data(dry_run=dry_run)
