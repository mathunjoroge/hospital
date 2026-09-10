#!/usr/bin/env python3
"""Clean up mock encounter data before stakeholder demo."""
from app import app
from extensions import db
from departments.models.encounter import Encounter

with app.app_context():
    count = Encounter.query.filter(
        Encounter.chief_complaint.like("Mock %")
    ).delete()
    db.session.commit()
    print(f"Deleted {count} mock encounters")
