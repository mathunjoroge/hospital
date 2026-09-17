"""
Kenya MOH-647 Facility Tracer Health Products and Technologies (HPT) Data Engine.

MOH 647: Health Facility Tracer Health Products and Technologies Data Report Form.
Tracks inventory levels, consumption (dispensed/issued), receipts, losses/expiries,
ending physical counts, and stockout statuses for essential Kenya MOH tracer commodities.
"""
from datetime import date, datetime, timedelta

from sqlalchemy import func, or_

from departments.models.pharmacy import DispensedDrug, Drug, Expiry, Purchase
from extensions import db

MOH647_TRACER_CATALOG = [
    {
        "category": "Antimicrobials",
        "name": "Amoxicillin 250mg / 500mg",
        "keywords": ["amoxicillin", "amoxil"],
        "unit": "Capsules / Suspension",
    },
    {
        "category": "Antimicrobials",
        "name": "Ceftriaxone 1g Injection",
        "keywords": ["ceftriaxone", "rocephin"],
        "unit": "Vials",
    },
    {
        "category": "Antimicrobials",
        "name": "Metronidazole 200mg / 400mg",
        "keywords": ["metronidazole", "flagyl"],
        "unit": "Tablets / IV",
    },
    {
        "category": "Antimicrobials",
        "name": "Ciprofloxacin 500mg",
        "keywords": ["ciprofloxacin", "cipro"],
        "unit": "Tablets",
    },
    {
        "category": "Analgesics",
        "name": "Paracetamol 500mg / Syrup",
        "keywords": ["paracetamol", "panadol", "acetaminophen"],
        "unit": "Tablets / Syrup",
    },
    {
        "category": "Analgesics",
        "name": "Ibuprofen 200mg / 400mg",
        "keywords": ["ibuprofen", "brufen"],
        "unit": "Tablets",
    },
    {
        "category": "Maternal & Reproductive",
        "name": "Oxytocin 10 IU Injection",
        "keywords": ["oxytocin", "pitocin"],
        "unit": "Ampoules",
    },
    {
        "category": "Maternal & Reproductive",
        "name": "Magnesium Sulphate 50% Inj",
        "keywords": ["magnesium sulphate", "magnesium sulfate", "mgso4"],
        "unit": "Ampoules",
    },
    {
        "category": "Maternal & Reproductive",
        "name": "Depo-Provera (Medroxyprogesterone)",
        "keywords": ["depo", "medroxyprogesterone"],
        "unit": "Vials",
    },
    {
        "category": "Child Health & Nutrition",
        "name": "ORS (Oral Rehydration Salts)",
        "keywords": ["ors", "oral rehydration"],
        "unit": "Sachets",
    },
    {
        "category": "Child Health & Nutrition",
        "name": "Zinc Sulphate 20mg",
        "keywords": ["zinc sulphate", "zinc sulfate", "zinc"],
        "unit": "Tablets",
    },
    {
        "category": "Child Health & Nutrition",
        "name": "Vitamin A 100,000 / 200,000 IU",
        "keywords": ["vitamin a", "retinol"],
        "unit": "Capsules",
    },
    {
        "category": "Malaria Commodities",
        "name": "Artemether-Lumefantrine (AL)",
        "keywords": ["artemether", "lumefantrine", "coartem", "al"],
        "unit": "Packs",
    },
    {
        "category": "Malaria Commodities",
        "name": "Artesunate 60mg Injection",
        "keywords": ["artesunate"],
        "unit": "Vials",
    },
    {
        "category": "NCDs & Chronic Care",
        "name": "Metformin 500mg / 850mg",
        "keywords": ["metformin", "glucophage"],
        "unit": "Tablets",
    },
    {
        "category": "NCDs & Chronic Care",
        "name": "Amlodipine 5mg / 10mg",
        "keywords": ["amlodipine", "norvasc"],
        "unit": "Tablets",
    },
    {
        "category": "NCDs & Chronic Care",
        "name": "Enalapril 5mg / 10mg",
        "keywords": ["enalapril", "renitec"],
        "unit": "Tablets",
    },
    {
        "category": "NCDs & Chronic Care",
        "name": "Salbutamol Inhaler 100mcg",
        "keywords": ["salbutamol", "ventolin"],
        "unit": "Inhalers",
    },
    {
        "category": "Diagnostics & Non-Pharm",
        "name": "Malaria RDT Kits",
        "keywords": ["malaria rdt", "rdt kit", "malaria rapid"],
        "unit": "Kits",
    },
    {
        "category": "Diagnostics & Non-Pharm",
        "name": "HIV Rapid Test Kits",
        "keywords": ["hiv rdt", "hiv rapid", "determine", "first response"],
        "unit": "Kits",
    },
    {
        "category": "Diagnostics & Non-Pharm",
        "name": "Surgical / Exam Gloves",
        "keywords": ["gloves", "latex gloves"],
        "unit": "Pairs / Boxes",
    },
]


def classify_tracer_item(generic_name: str) -> dict | None:
    """
    Match a drug or non-pharm generic name against the Kenya MOH 647 Tracer Catalog.
    Returns the matching catalog dictionary or None.
    """
    if not generic_name:
        return None

    name_lower = generic_name.lower()
    for tracer in MOH647_TRACER_CATALOG:
        for kw in tracer["keywords"]:
            if kw in name_lower:
                return tracer
    return None


def aggregate_moh647_monthly(year: int = None, month: int = None) -> dict:
    """
    Aggregate monthly inventory, consumption, stockouts, and expiries for MOH 647.
    """
    today = date.today()
    if year is None:
        year = today.year
    if month is None:
        month = today.month

    start_date = date(year, month, 1)
    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    # Get all drugs in the system
    all_drugs = Drug.query.all()

    tracer_reports = []
    total_monitored = 0
    in_stock_count = 0
    stockout_count = 0
    low_stock_count = 0

    # Group drugs by matching tracer category
    mapped_tracer_names = set()

    for drug in all_drugs:
        catalog_match = classify_tracer_item(drug.generic_name)
        if not catalog_match:
            continue

        tracer_key = catalog_match["name"]
        mapped_tracer_names.add(tracer_key)

        # 1. Received in month
        received = (
            db.session.query(func.coalesce(func.sum(Purchase.quantity_purchased), 0))
            .filter(Purchase.drug_id == drug.id)
            .filter(Purchase.purchase_date >= start_date)
            .filter(Purchase.purchase_date <= end_date)
            .scalar()
        )

        # 2. Dispensed / Issued in month
        issued = (
            db.session.query(func.coalesce(func.sum(DispensedDrug.quantity_dispensed), 0))
            .filter(DispensedDrug.drug_id == drug.id)
            .filter(DispensedDrug.date_dispensed >= start_dt)
            .filter(DispensedDrug.date_dispensed <= end_dt)
            .filter(or_(DispensedDrug.status == "0", DispensedDrug.status == "COMPLETED", DispensedDrug.status == "DISPENSED", DispensedDrug.status.is_(None)))
            .scalar()
        )

        # 3. Losses / Expiries in month
        losses = (
            db.session.query(func.coalesce(func.sum(Expiry.quantity_removed), 0))
            .filter(Expiry.drug_id == drug.id)
            .filter(Expiry.removal_date >= start_date)
            .filter(Expiry.removal_date <= end_date)
            .scalar()
        )

        # 4. Current ending balance
        ending_stock = drug.quantity_in_stock or 0
        reorder = drug.reorder_level or 0

        # Derived beginning stock approximation
        beginning_stock = max(0, ending_stock + issued + losses - received)

        is_stockout = ending_stock == 0
        is_low = ending_stock > 0 and ending_stock <= reorder

        total_monitored += 1
        if is_stockout:
            stockout_count += 1
        elif is_low:
            low_stock_count += 1
            in_stock_count += 1
        else:
            in_stock_count += 1

        status_label = "STOCKOUT" if is_stockout else ("LOW STOCK" if is_low else "IN STOCK")

        tracer_reports.append(
            {
                "drug_id": drug.id,
                "generic_name": drug.generic_name,
                "brand_name": drug.brand_name or drug.generic_name,
                "tracer_name": catalog_match["name"],
                "category": catalog_match["category"],
                "unit": catalog_match["unit"],
                "beginning_stock": beginning_stock,
                "received": int(received),
                "issued": int(issued),
                "losses": int(losses),
                "ending_stock": ending_stock,
                "reorder_level": reorder,
                "status": status_label,
                "is_stockout": is_stockout,
                "is_low_stock": is_low,
            }
        )

    # Sort reports by category then generic name
    tracer_reports.sort(key=lambda x: (x["category"], x["generic_name"]))

    return {
        "year": year,
        "month": month,
        "total_monitored": total_monitored,
        "in_stock_count": in_stock_count,
        "stockout_count": stockout_count,
        "low_stock_count": low_stock_count,
        "tracer_items": tracer_reports,
    }
