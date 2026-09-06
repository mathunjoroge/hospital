"""
generate_product_pdf.py
Generates docs/HMIS_Product_Overview.pdf using ReportLab.
Run: python scripts/generate_product_pdf.py
"""
import os
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── Colour palette (matches the teal-blue UI theme) ──────────────────────────
TEAL_DEEP   = colors.HexColor("#164e63")
TEAL_MID    = colors.HexColor("#0e7490")
TEAL_LIGHT  = colors.HexColor("#0891b2")
CYAN_PALE   = colors.HexColor("#ecfeff")
SLATE_700   = colors.HexColor("#334155")
SLATE_500   = colors.HexColor("#64748b")
SLATE_100   = colors.HexColor("#f1f5f9")
WHITE       = colors.white
ACCENT_GOLD = colors.HexColor("#f59e0b")

W, H = A4

# ── Styles ────────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    cover_title = ParagraphStyle(
        "CoverTitle",
        fontSize=34,
        fontName="Helvetica-Bold",
        textColor=WHITE,
        leading=40,
        spaceAfter=6,
    )
    cover_sub = ParagraphStyle(
        "CoverSub",
        fontSize=14,
        fontName="Helvetica",
        textColor=colors.HexColor("#a5f3fc"),
        leading=20,
        spaceAfter=4,
    )
    cover_date = ParagraphStyle(
        "CoverDate",
        fontSize=10,
        fontName="Helvetica",
        textColor=colors.HexColor("#cffafe"),
        leading=14,
    )
    section_h = ParagraphStyle(
        "SectionH",
        fontSize=16,
        fontName="Helvetica-Bold",
        textColor=TEAL_DEEP,
        spaceBefore=18,
        spaceAfter=6,
        leading=20,
    )
    sub_h = ParagraphStyle(
        "SubH",
        fontSize=12,
        fontName="Helvetica-Bold",
        textColor=TEAL_MID,
        spaceBefore=10,
        spaceAfter=4,
        leading=16,
    )
    body = ParagraphStyle(
        "Body",
        fontSize=10,
        fontName="Helvetica",
        textColor=SLATE_700,
        leading=16,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
    )
    bullet = ParagraphStyle(
        "Bullet",
        fontSize=10,
        fontName="Helvetica",
        textColor=SLATE_700,
        leading=15,
        leftIndent=14,
        spaceAfter=3,
    )
    caption = ParagraphStyle(
        "Caption",
        fontSize=8,
        fontName="Helvetica-Oblique",
        textColor=SLATE_500,
        alignment=TA_CENTER,
    )
    kpi_val = ParagraphStyle(
        "KpiVal",
        fontSize=22,
        fontName="Helvetica-Bold",
        textColor=TEAL_LIGHT,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    kpi_lbl = ParagraphStyle(
        "KpiLbl",
        fontSize=9,
        fontName="Helvetica",
        textColor=SLATE_500,
        alignment=TA_CENTER,
    )
    return dict(
        cover_title=cover_title, cover_sub=cover_sub, cover_date=cover_date,
        section_h=section_h, sub_h=sub_h, body=body, bullet=bullet,
        caption=caption, kpi_val=kpi_val, kpi_lbl=kpi_lbl,
    )


# ── Page templates ────────────────────────────────────────────────────────────
def cover_bg(canvas, doc):
    """Full-bleed teal gradient cover background."""
    canvas.saveState()
    # Background rectangle (gradient approximated with two rects)
    canvas.setFillColor(TEAL_DEEP)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    canvas.setFillColor(TEAL_MID)
    canvas.rect(0, 0, W, H * 0.55, fill=1, stroke=0)
    # Decorative circle
    canvas.setFillColor(colors.HexColor("#0e7490"))
    canvas.setStrokeColor(colors.HexColor("#0891b2"))
    canvas.setLineWidth(0)
    canvas.circle(W - 60, H - 60, 120, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#164e63"))
    canvas.circle(W - 60, H - 60, 80, fill=1, stroke=0)
    # Bottom accent bar
    canvas.setFillColor(ACCENT_GOLD)
    canvas.rect(0, 30 * mm, W, 3, fill=1, stroke=0)
    canvas.restoreState()


def normal_header_footer(canvas, doc):
    """Header/footer for inner pages."""
    canvas.saveState()
    # Header bar
    canvas.setFillColor(TEAL_DEEP)
    canvas.rect(0, H - 18 * mm, W, 18 * mm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(20 * mm, H - 11 * mm, "HMIS — Hospital Management Information System")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(W - 20 * mm, H - 11 * mm, "Product Overview  |  Confidential")
    # Footer
    canvas.setFillColor(SLATE_100)
    canvas.rect(0, 0, W, 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(TEAL_MID)
    canvas.rect(0, 0, W, 1.5, fill=1, stroke=0)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SLATE_500)
    canvas.drawString(20 * mm, 4 * mm, f"© {date.today().year}  —  All rights reserved")
    canvas.drawRightString(W - 20 * mm, 4 * mm, f"Page {doc.page}")
    canvas.restoreState()


# ── Helper builders ───────────────────────────────────────────────────────────
def hr(color=TEAL_LIGHT, thickness=0.8):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=2)


def kpi_table(s, data):
    """Renders a row of KPI boxes: [(value, label), ...]"""
    cells = []
    for val, lbl in data:
        cells.append([
            Paragraph(val, s["kpi_val"]),
            Paragraph(lbl, s["kpi_lbl"]),
        ])
    tbl = Table(
        [[[Paragraph(v, s["kpi_val"]), Paragraph(l, s["kpi_lbl"])] for v, l in data]],
        colWidths=[(W - 60 * mm) / len(data)] * len(data),
    )
    tbl.setStyle(TableStyle([
        ("BOX",        (0, 0), (-1, -1), 0.5, TEAL_LIGHT),
        ("LINEAFTER",  (0, 0), (-2, -1), 0.5, colors.HexColor("#a5f3fc")),
        ("BACKGROUND", (0, 0), (-1, -1), CYAN_PALE),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [6]),
    ]))
    return tbl


def feature_table(s, rows, col_widths=None):
    """Two-column feature table with teal header row."""
    col_widths = col_widths or [6 * cm, 10.5 * cm]
    header = [
        Paragraph("<b>Module</b>", ParagraphStyle("th", fontSize=10, fontName="Helvetica-Bold", textColor=WHITE)),
        Paragraph("<b>Key Capabilities</b>", ParagraphStyle("th", fontSize=10, fontName="Helvetica-Bold", textColor=WHITE)),
    ]
    tdata = [header] + [
        [Paragraph(f"<b>{r[0]}</b>", ParagraphStyle("td1", fontSize=9, fontName="Helvetica-Bold", textColor=TEAL_DEEP, leading=13)),
         Paragraph(r[1], ParagraphStyle("td2", fontSize=9, fontName="Helvetica", textColor=SLATE_700, leading=13))]
        for r in rows
    ]
    tbl = Table(tdata, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  TEAL_DEEP),
        ("BACKGROUND",    (0, 1), (-1, -1), WHITE),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, SLATE_100]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("TOPPADDING",    (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return tbl


# ── Document content ──────────────────────────────────────────────────────────
def build_story(s):
    story = []

    # ── COVER ──────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 6 * cm))
    story.append(Paragraph("HMIS", s["cover_title"]))
    story.append(Paragraph("Hospital Management Information System", s["cover_sub"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Product Overview &amp; Capability Reference", s["cover_sub"]))
    story.append(Spacer(1, 14 * mm))
    story.append(Paragraph(f"Version 1.0  ·  {date.today().strftime('%B %Y')}  ·  Confidential", s["cover_date"]))
    story.append(PageBreak())

    # ── Switch to inner template ───────────────────────────────────────────────
    story.append(NextPageTemplate("inner"))

    # ── 1. EXECUTIVE SUMMARY ──────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", s["section_h"]))
    story.append(hr())
    story.append(Paragraph(
        "The Hospital Management Information System (HMIS) is a full-stack, cloud-ready "
        "clinical and administrative platform engineered specifically for East African healthcare "
        "facilities. It digitises every touchpoint of a patient's journey — from triage and "
        "outpatient consultation through inpatient ward rounds, pharmacy dispensing, laboratory "
        "results, imaging review, billing, and discharge — while remaining compliant with Kenya's "
        "Data Protection Act 2019 and connected to national reporting infrastructure (DHIS2/KHIS "
        "and the SHA/SHIF insurance scheme).",
        s["body"]))
    story.append(Paragraph(
        "Built on a modern, containerised Python/Flask architecture backed by PostgreSQL, the "
        "system has been hardened through six development phases, stress-tested at sustained "
        "concurrency, and prepared for independent clinical safety and security audits. With "
        "214 automated tests, a 41 % code-coverage baseline, and a 0.0 % error rate on all "
        "hot-path benchmarks, the platform is engineered to the standard required before "
        "admitting the first real patient.",
        s["body"]))

    story.append(Spacer(1, 6 * mm))
    story.append(kpi_table(s, [
        ("214", "Automated tests\npassing"),
        ("41.6 %", "Code coverage\nbaseline"),
        ("0.0 %", "Error rate on\nhot-path load tests"),
        ("17", "Clinical &amp; admin\nmodules"),
        ("6", "Development\nphases completed"),
    ]))
    story.append(Spacer(1, 8 * mm))

    # ── 2. CLINICAL MODULES ───────────────────────────────────────────────────
    story.append(Paragraph("2. Clinical Modules", s["section_h"]))
    story.append(hr())
    story.append(feature_table(s, [
        ("Outpatient / OPD",
         "Patient registration with UUID, triage vitals capture, OPD waiting-list queue, "
         "SOAP/SBAR consultation notes, ICD-10 diagnosis coding (50+ curated codes), "
         "AI Clinical Assistant (chatbot with audit trail &amp; consent gate)."),
        ("Ward Rounds &amp; Inpatients",
         "Admit/discharge workflow, ward-bed history, daily ward round notes, "
         "Medication Administration Record (MAR), nurse notification system."),
        ("Prescribing &amp; CDSS",
         "Drug prescribing with Clinical Decision Support: 7,621 live drug–drug interaction "
         "rules via DrugCentral PostgreSQL, allergen class screening (5 classes), renal dose "
         "adjustment, Patient Global Allergy Registry, Active Problem List."),
        ("Pharmacy",
         "Dispensing workflow, FEFO (First-Expiry-First-Out) batch inventory, automated "
         "low-stock scanning, auto-generated Supplier Purchase Orders, AI drug-discovery "
         "assistant, cheminformatics fingerprint search."),
        ("Laboratory",
         "Lab test ordering from consultation, LIS result entry, panic-value alert system "
         "(LIS blueprint), result gating before patient-portal release."),
        ("Imaging / Radiology",
         "DICOM study upload (up to 2 GB), Cornerstone.js in-browser viewer, "
         "unmatched-imaging reconciliation queue, PACS/HL7 interfacing research documented."),
        ("Theatre &amp; Oncology",
         "Theatre booking and list management, post-operative note update, "
         "Oncology clinic with chemotherapy regimen tracking, AI treatment summary "
         "(consent-gated)."),
        ("Emergency Access",
         "Break-glass emergency override with full audit trail, time-limited access tokens, "
         "supervisor alert dispatching, admin audit-trail view."),
        ("Telemedicine",
         "WebRTC virtual consultation room with real-time signalling via Socket.IO, "
         "in-call clinical notes, prescription drafting. Feature-flagged "
         "(ENABLE_TELEMEDICINE) pending regulatory sign-off."),
    ]))
    story.append(PageBreak())

    # ── 3. ADMINISTRATIVE & OPERATIONAL MODULES ───────────────────────────────
    story.append(Paragraph("3. Administrative &amp; Operational Modules", s["section_h"]))
    story.append(hr())
    story.append(feature_table(s, [
        ("Billing &amp; Finance",
         "Invoice generation, multi-payment allocation (cash, M-Pesa STK Push, insurance), "
         "unreconciled-charges aggregation, revenue analytics."),
        ("M-Pesa Integration",
         "Safaricom Daraja API STK Push and C2B callback handling, payment receipting, "
         "patient-portal self-pay flow."),
        ("SHA/SHIF Insurance",
         "InsuranceScheme &amp; PatientInsurance models, claim adjudication workflow, "
         "approval/rejection tracking, analytics breakdown."),
        ("HR &amp; Credentialing",
         "Staff roster management, professional-licence upload, expiry-date warning alerts, "
         "StaffCredential model with admin view."),
        ("Stores &amp; Inventory",
         "Central stores issue/return workflow, stock-level tracking, reorder triggers."),
        ("Mortuary",
         "Deceased-patient registration, body release workflow."),
        ("Analytics Dashboard",
         "Executive dashboard (Chart.js): bed occupancy trend, 30-day admission curve, "
         "revenue breakdown by payment channel, insurance claim approval ratios. "
         "JSON API for BI tool integration."),
        ("Patient Self-Service Portal",
         "Separate PatientUser authentication, appointment booking, lab-result viewing "
         "(gated), billing history, M-Pesa self-pay, profile management with audit log."),
        ("Outbound Communications",
         "Flask-Mail email driver, SMS sandbox abstraction, OutboundNotificationLog, "
         "5 event triggers (appointments, labs, billing, payments, claims), "
         "24-hour appointment reminder scheduler."),
        ("Audit &amp; Logging",
         "AuditLog DB model, @audited decorator on all write routes, SIEM-export endpoint, "
         "full structured event log with user/IP/timestamp/diff."),
    ]))
    story.append(PageBreak())

    # ── 4. INTEROPERABILITY ───────────────────────────────────────────────────
    story.append(Paragraph("4. Interoperability &amp; Standards", s["section_h"]))
    story.append(hr())
    story.append(feature_table(s, [
        ("FHIR R4 API",
         "RESTful /api/fhir/R4 endpoints for Patient, Observation, MedicationRequest, "
         "DiagnosticReport resources. Enables integration with national health exchanges."),
        ("DHIS2 / KHIS Export",
         "/api/khis blueprint: automated aggregate report generation and push to Kenya's "
         "national DHIS2 instance for MOH reporting compliance."),
        ("JWT REST API",
         "Bearer-token authenticated /api/* endpoints for mobile clients, "
         "third-party EHR connectors, and BI dashboards."),
        ("M-Pesa (Daraja)",
         "STK Push initiation and C2B webhook for real-time payment reconciliation."),
        ("ICD-10 Coding",
         "50+ curated ICD-10-CM codes across all departments; WHO API credential "
         "requirement documented for full 70,000-code database ingestion."),
        ("DICOM / PACS",
         "DICOM Web upload &amp; Cornerstone.js viewer. HL7 MLLP/ASTM LIS interfacing "
         "architecture documented; implementation pending vendor selection."),
    ]))
    story.append(Spacer(1, 8 * mm))

    # ── 5. SECURITY & COMPLIANCE ──────────────────────────────────────────────
    story.append(Paragraph("5. Security &amp; Compliance", s["section_h"]))
    story.append(hr())
    story.append(Paragraph(
        "Security is implemented as a layered defence-in-depth strategy, not an afterthought:", s["body"]))

    sec_items = [
        ("Authentication", "TOTP-based MFA, 5-attempt account lockout (15-minute timeout), "
         "bcrypt/PBKDF2 password hashing, session signed with SECRET_KEY."),
        ("Authorisation", "Role-based access control (RBAC) across 12 roles; "
         "@break_glass_required decorator for emergency overrides with full audit trail."),
        ("Encryption at Rest", "Fernet AES-128-CBC + HMAC EncryptedString column type "
         "applied to all patient PII identity fields."),
        ("Data Protection Act 2019", "PatientConsent model, Subject Access Request JSON export, "
         "patient anonymisation route, data-residency hard stop documented."),
        ("CSRF Protection", "Flask-WTF CSRF tokens on all state-changing forms; "
         "JWT API endpoints explicitly exempted."),
        ("Rate Limiting", "Flask-Limiter on login (5/min POST), disabled in test mode."),
        ("CI Security Gates", "pip-audit (17 known advisories documented), bandit SAST, "
         "import-order linting — all enforced in GitHub Actions without || true bypass."),
        ("Session Security", "Redis-backed sessions, 30-minute idle timeout, SameSite=Lax, "
         "HttpOnly, Secure cookie flags."),
    ]
    story.append(feature_table(s, sec_items, col_widths=[4.5 * cm, 12 * cm]))
    story.append(PageBreak())

    # ── 6. ARCHITECTURE ────────────────────────────────────────────────────────
    story.append(Paragraph("6. Technical Architecture", s["section_h"]))
    story.append(hr())

    story.append(Paragraph("6.1  Stack Overview", s["sub_h"]))
    story.append(feature_table(s, [
        ("Backend",       "Python 3.12 · Flask 3.x · SQLAlchemy ORM · Flask-Migrate (Alembic)"),
        ("Database",      "PostgreSQL 16 (primary) · SQLite (test / offline fallback)"),
        ("Cache / Queue", "Redis 7 (session store + Celery broker) · Celery 5.4 (async tasks)"),
        ("Real-time",     "Flask-SocketIO · WebRTC (telemedicine signalling)"),
        ("Auth",          "Flask-Login · Flask-JWT-Extended · PyOTP (TOTP MFA)"),
        ("AI / NLP",      "NVIDIA NIM API client · local HuggingFace summariser · DrugCentral DDI DB"),
        ("Containers",    "Docker + Docker Compose (web, db, redis, celery_worker, nlp services)"),
        ("CI/CD",         "GitHub Actions: lint → bandit → pip-audit → pytest (214 tests, 41.6 % cov)"),
        ("Offline / PWA", "Service Worker sw.js · Web App Manifest · offline.html fallback"),
    ], col_widths=[4 * cm, 12.5 * cm]))

    story.append(Paragraph("6.2  Deployment", s["sub_h"]))
    story.append(Paragraph(
        "The system ships as a five-container Docker Compose stack. A single "
        "<b>docker compose up -d</b> command starts the Flask application server, "
        "PostgreSQL database, Redis broker, Celery worker, and the optional NLP micro-service. "
        "Database schema changes are applied with <b>flask db upgrade</b> (Alembic). "
        "The stack is cloud-agnostic and has been tested on bare-metal Ubuntu 22.04 LTS and "
        "AWS EC2/RDS equivalents.", s["body"]))

    story.append(Paragraph("6.3  Performance Baseline", s["sub_h"]))
    story.append(Paragraph(
        "Hot-path benchmarking was executed with a session-authenticated concurrent load "
        "test (C = 20 simultaneous workers). Results recorded in docs/load_test_results.md:", s["body"]))
    story.append(feature_table(s, [
        ("Patient Registration",   "0.0 % error rate · sub-200 ms p95 at C=20"),
        ("Prescription Sign-off",  "0.0 % error rate · sub-250 ms p95 at C=20"),
        ("Billing Payment",        "0.0 % error rate · sub-200 ms p95 at C=20"),
        ("Health Check /healthz",  "1,200 req/sec sustained · Flask-Limiter lockout verified"),
    ], col_widths=[6 * cm, 10.5 * cm]))
    story.append(PageBreak())

    # ── 7. COMPLIANCE & AUDIT READINESS ───────────────────────────────────────
    story.append(Paragraph("7. Audit &amp; Regulatory Readiness", s["section_h"]))
    story.append(hr())
    story.append(Paragraph(
        "Three independent audit-preparation packages have been produced and maintained "
        "in the docs/ directory:", s["body"]))
    for doc_name, desc in [
        ("security_audit_readiness.md",
         "Maps every OWASP Top-10 2021 control to its implementation in the codebase. "
         "Identifies 17 open pip-audit advisories with risk ratings and mitigations."),
        ("clinical_safety_review_packaging.md",
         "Documents the Clinical Decision Support rules, AI governance framework "
         "(input validation, consent gating, mode disclosure, audit timer), "
         "allergy registry, and known limitations for clinical safety reviewers."),
        ("accessibility_audit_report.md",
         "WCAG 2.1 AA compliance checklist. Documents keyboard navigation, ARIA landmark "
         "roles, colour-contrast ratios, and screen-reader compatibility across core flows."),
    ]:
        story.append(Paragraph(f"<b>docs/{doc_name}</b>", s["sub_h"]))
        story.append(Paragraph(desc, s["body"]))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "A DECISIONS_PENDING.md register captures all items requiring regulatory, financial, "
        "or clinical authority before the system goes live with real patients (controlled drug "
        "register, eTIMS/KRA tax compliance, file-storage data-residency, WHO ICD-10 licence).",
        s["body"]))
    story.append(PageBreak())

    # ── 8. ROADMAP ─────────────────────────────────────────────────────────────
    story.append(Paragraph("8. Recommended Next Steps", s["section_h"]))
    story.append(hr())
    story.append(Paragraph(
        "The following items have been identified as the highest-value investments to elevate "
        "the platform from a production-ready Level 2 HMIS to a world-class enterprise system:", s["body"]))

    roadmap = [
        ("ICD-10 Full Database",
         "Replace the 50-item curated array in prescribe.py with a full ICD-10-CM SQLite "
         "table and FTS5/Trigram search index (requires WHO API licence)."),
        ("KRA eTIMS Middleware",
         "Implement sign-and-send middleware for electronic tax invoicing compliance "
         "with the Kenya Revenue Authority (pending regulatory decision)."),
        ("Controlled Drug Register",
         "Dedicated Schedule I/II/III dispensing log, dual-signature workflow, and "
         "monthly reconciliation report (pending Pharmacy &amp; Poisons Board guidance)."),
        ("LIS / PACS Hardware Bridge",
         "ASTM E1381 MLLP listener for lab analyser auto-result import and DICOM "
         "worklist integration with physical imaging equipment."),
        ("HL7 ADT Feed",
         "Real-time Admit/Discharge/Transfer HL7 v2 feed to national health exchange."),
        ("Telemedicine Activation",
         "Complete KMPDC telehealth guideline review and activate the ENABLE_TELEMEDICINE "
         "feature flag for video consultations."),
        ("Penetration Test",
         "Commission a third-party OWASP-scoped web application penetration test before "
         "production go-live."),
    ]
    story.append(feature_table(s, roadmap, col_widths=[5.5 * cm, 11 * cm]))
    story.append(Spacer(1, 10 * mm))

    # ── 9. CONTACT / CLOSING ──────────────────────────────────────────────────
    story.append(Paragraph("9. Contact &amp; Licensing", s["section_h"]))
    story.append(hr())
    story.append(Paragraph(
        "This document is confidential and intended solely for the recipient. "
        "The HMIS codebase is maintained under a proprietary licence. "
        "For partnership enquiries, clinical deployment support, or integration "
        "services, please contact the development team through the project repository.",
        s["body"]))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        f"Document generated: {date.today().strftime('%d %B %Y')}",
        s["caption"]))

    return story


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "HMIS_Product_Overview.pdf")
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    doc = BaseDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=25 * mm,
        bottomMargin=20 * mm,
        title="HMIS — Hospital Management Information System",
        author="HMIS Development Team",
        subject="Product Overview & Capability Reference",
    )

    # Cover template (no header/footer, full bleed)
    cover_frame = Frame(0, 0, W, H, leftPadding=30 * mm, rightPadding=30 * mm,
                        topPadding=0, bottomPadding=30 * mm, id="cover")
    cover_tpl = PageTemplate(id="cover", frames=[cover_frame], onPage=cover_bg)

    # Inner pages
    inner_frame = Frame(
        20 * mm, 18 * mm,
        W - 40 * mm, H - 18 * mm - 20 * mm,
        id="inner_body",
    )
    inner_tpl = PageTemplate(id="inner", frames=[inner_frame], onPage=normal_header_footer)

    doc.addPageTemplates([cover_tpl, inner_tpl])

    s = make_styles()
    story = build_story(s)
    doc.build(story)
    print(f"✅  PDF written to: {out_path}")


if __name__ == "__main__":
    main()
