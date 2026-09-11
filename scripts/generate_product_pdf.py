#!/usr/bin/env python3
"""
generate_product_pdf.py
Generates docs/HMIS_Product_Overview.pdf using ReportLab.

Usage:
    python scripts/generate_product_pdf.py
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── Color Palette (Teal / Slate Theme) ────────────────────────────────────────
TEAL_DEEP = colors.HexColor("#164e63")
TEAL_MID = colors.HexColor("#0e7490")
TEAL_LIGHT = colors.HexColor("#0891b2")
CYAN_PALE = colors.HexColor("#ecfeff")
SLATE_700 = colors.HexColor("#0f172a")  # Primary body text for print
SLATE_500 = colors.HexColor("#64748b")
SLATE_100 = colors.HexColor("#f1f5f9")
WHITE = colors.white
ACCENT_GOLD = colors.HexColor("#f59e0b")

PAGE_WIDTH, PAGE_HEIGHT = A4


# ── Two-Pass Canvas for Dynamic Total Page Numbers ────────────────────────────
class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute total page counts (e.g., 'Page X of Y').
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int) -> None:
        if self._pageNumber == 1:
            # Draw Cover Page Background
            self.saveState()
            self.setFillColor(TEAL_DEEP)
            self.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
            self.setFillColor(TEAL_MID)
            self.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT * 0.55, fill=1, stroke=0)
            # Decorative Geometry
            self.setFillColor(colors.HexColor("#0e7490"))
            self.circle(PAGE_WIDTH - 60, PAGE_HEIGHT - 60, 120, fill=1, stroke=0)
            self.setFillColor(colors.HexColor("#164e63"))
            self.circle(PAGE_WIDTH - 60, PAGE_HEIGHT - 60, 80, fill=1, stroke=0)
            # Accent Stripe
            self.setFillColor(ACCENT_GOLD)
            self.rect(0, 30 * mm, PAGE_WIDTH, 3, fill=1, stroke=0)
            self.restoreState()
            return

        # Header / Footer for Inner Pages
        self.saveState()
        # Running Header Bar
        self.setFillColor(TEAL_DEEP)
        self.rect(0, PAGE_HEIGHT - 18 * mm, PAGE_WIDTH, 18 * mm, fill=1, stroke=0)
        self.setFillColor(WHITE)
        self.setFont("Helvetica-Bold", 9)
        self.drawString(20 * mm, PAGE_HEIGHT - 11 * mm, "HMIS — Hospital Management Information System")
        self.setFont("Helvetica", 8)
        self.drawRightString(PAGE_WIDTH - 20 * mm, PAGE_HEIGHT - 11 * mm, "Product Overview  |  Confidential")

        # Running Footer Bar
        self.setFillColor(SLATE_100)
        self.rect(0, 0, PAGE_WIDTH, 12 * mm, fill=1, stroke=0)
        self.setFillColor(TEAL_MID)
        self.rect(0, 0, PAGE_WIDTH, 1.5, fill=1, stroke=0)
        self.setFont("Helvetica", 8)
        self.setFillColor(SLATE_500)
        self.drawString(20 * mm, 4 * mm, f"© {datetime.now(timezone.utc).date().year} — All rights reserved")
        self.drawRightString(PAGE_WIDTH - 20 * mm, 4 * mm, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


# ── Styles ────────────────────────────────────────────────────────────────────
def make_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()

    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=styles["Normal"],
            fontSize=34,
            fontName="Helvetica-Bold",
            textColor=WHITE,
            leading=40,
            spaceAfter=6,
        ),
        "cover_sub": ParagraphStyle(
            "CoverSub",
            parent=styles["Normal"],
            fontSize=14,
            fontName="Helvetica",
            textColor=colors.HexColor("#a5f3fc"),
            leading=20,
            spaceAfter=4,
        ),
        "cover_date": ParagraphStyle(
            "CoverDate",
            parent=styles["Normal"],
            fontSize=10,
            fontName="Helvetica",
            textColor=colors.HexColor("#cffafe"),
            leading=14,
        ),
        "section_h": ParagraphStyle(
            "SectionH",
            parent=styles["Normal"],
            fontSize=16,
            fontName="Helvetica-Bold",
            textColor=TEAL_DEEP,
            spaceBefore=16,
            spaceAfter=6,
            leading=20,
            keepWithNext=True,
        ),
        "sub_h": ParagraphStyle(
            "SubH",
            parent=styles["Normal"],
            fontSize=11,
            fontName="Helvetica-Bold",
            textColor=TEAL_MID,
            spaceBefore=10,
            spaceAfter=4,
            leading=15,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=9.5,
            fontName="Helvetica",
            textColor=SLATE_700,
            leading=15,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontSize=8,
            fontName="Helvetica-Oblique",
            textColor=SLATE_500,
            alignment=TA_CENTER,
        ),
        "kpi_val": ParagraphStyle(
            "KpiVal",
            parent=styles["Normal"],
            fontSize=16,
            fontName="Helvetica-Bold",
            textColor=TEAL_LIGHT,
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "kpi_lbl": ParagraphStyle(
            "KpiLbl",
            parent=styles["Normal"],
            fontSize=8.5,
            fontName="Helvetica",
            textColor=SLATE_500,
            alignment=TA_CENTER,
            leading=11,
        ),
    }


# ── Helper Component Builders ─────────────────────────────────────────────────
def create_hr(color: colors.Color = TEAL_LIGHT, thickness: float = 0.8) -> HRFlowable:
    return HRFlowable(
        width="100%",
        thickness=thickness,
        color=color,
        spaceAfter=8,
        spaceBefore=2,
        hAlign="CENTER",
    )


def build_kpi_table(styles: dict[str, ParagraphStyle], data: list[tuple[str, str]]) -> Table:
    """Renders a row of metric key indicators evenly across the content width."""
    content_width = PAGE_WIDTH - 40 * mm
    col_w = content_width / len(data)

    formatted_data = [
        [[Paragraph(val, styles["kpi_val"]), Paragraph(lbl, styles["kpi_lbl"])] for val, lbl in data]
    ]

    tbl = Table(formatted_data, colWidths=[col_w] * len(data))
    tbl.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, TEAL_LIGHT),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#a5f3fc")),
        ("BACKGROUND", (0, 0), (-1, -1), CYAN_PALE),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def build_feature_table(
    styles: dict[str, ParagraphStyle],
    rows: list[tuple[str, str]],
    col_widths: list[float] | None = None,
) -> Table:
    """Renders a two-column module and capabilities data grid."""
    col_widths = col_widths or [5.5 * cm, 11.5 * cm]

    header_col1 = Paragraph("<b>Module</b>", ParagraphStyle("TH1", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE))
    header_col2 = Paragraph("<b>Key Capabilities</b>", ParagraphStyle("TH2", fontSize=9, fontName="Helvetica-Bold", textColor=WHITE))

    tdata = [[header_col1, header_col2]]

    td_label_style = ParagraphStyle("TDLabel", fontSize=8.5, fontName="Helvetica-Bold", textColor=TEAL_DEEP, leading=12)
    td_body_style = ParagraphStyle("TDBody", fontSize=8.5, fontName="Helvetica", textColor=SLATE_700, leading=12)

    for label, desc in rows:
        tdata.append([
            Paragraph(label, td_label_style),
            Paragraph(desc, td_body_style),
        ])

    tbl = Table(tdata, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL_DEEP),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SLATE_100]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return tbl


# ── Story Construction ────────────────────────────────────────────────────────
def build_story(styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    story: list[Flowable] = []

    # Cover Page
    story.append(Spacer(1, 6 * cm))
    story.append(Paragraph("HMIS", styles["cover_title"]))
    story.append(Paragraph("Hospital Management Information System", styles["cover_sub"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Product Overview &amp; Capability Reference", styles["cover_sub"]))
    story.append(Spacer(1, 14 * mm))
    story.append(Paragraph(f"Version 1.0  ·  {datetime.now(timezone.utc).date().strftime('%B %Y')}  ·  Confidential", styles["cover_date"]))
    story.append(PageBreak())

    # Inner Pages Switch
    story.append(NextPageTemplate("inner"))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", styles["section_h"]))
    story.append(create_hr())
    story.append(Paragraph(
        "The Hospital Management Information System (HMIS) is a full-stack, cloud-ready "
        "clinical and administrative platform engineered specifically for East African healthcare "
        "facilities. It digitises every touchpoint of a patient's journey — from triage and "
        "outpatient consultation through inpatient ward rounds, pharmacy dispensing, laboratory "
        "results, imaging review, billing, and discharge — while remaining compliant with Kenya's "
        "Data Protection Act 2019 and connected to national reporting infrastructure (DHIS2/KHIS "
        "and the SHA/SHIF insurance scheme).",
        styles["body"],
    ))
    story.append(Paragraph(
        "Built on a modern, containerised Python/Flask architecture backed by PostgreSQL, the "
        "system has been hardened through six development phases, stress-tested at sustained "
        "concurrency, and prepared for independent clinical safety and security audits. With "
        "214 automated tests, a 41.6% code-coverage baseline, and a 0.0% error rate on all "
        "hot-path benchmarks, the platform is engineered to enterprise reliability standards.",
        styles["body"],
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(build_kpi_table(styles, [
        ("214", "Automated tests passing"),
        ("41.6%", "Code coverage baseline"),
        ("0.0%", "Hot-path error rate"),
        ("17", "Clinical & admin modules"),
        ("6", "Phases completed"),
    ]))
    story.append(Spacer(1, 6 * mm))

    # 2. Clinical Modules
    story.append(Paragraph("2. Clinical Modules", styles["section_h"]))
    story.append(create_hr())
    story.append(build_feature_table(styles, [
        ("Outpatient / OPD",
         ("Patient registration with UUID, triage vitals capture, OPD waiting-list queue, "
         "SOAP/SBAR consultation notes, ICD-10 diagnosis coding (50+ curated codes), "
         "AI Clinical Assistant (chatbot with audit trail &amp; consent gate).")),
        ("Ward Rounds &amp; Inpatients",
         ("Admit/discharge workflow, ward-bed history, daily ward round notes, "
         "Medication Administration Record (MAR), nurse notification system.")),
        ("Prescribing &amp; CDSS",
         ("Drug prescribing with Clinical Decision Support: 7,621 live drug–drug interaction "
         "rules via DrugCentral PostgreSQL, allergen class screening (5 classes), renal dose "
         "adjustment, Patient Global Allergy Registry, Active Problem List.")),
        ("Pharmacy",
         ("Dispensing workflow, FEFO (First-Expiry-First-Out) batch inventory, automated "
         "low-stock scanning, auto-generated Supplier Purchase Orders, AI drug-discovery "
         "assistant, cheminformatics fingerprint search.")),
        ("Laboratory",
         ("Lab test ordering from consultation, LIS result entry, panic-value alert system "
         "(LIS blueprint), result gating before patient-portal release.")),
        ("Imaging / Radiology",
         ("DICOM study upload (up to 2 GB), Cornerstone.js in-browser viewer, "
         "unmatched-imaging reconciliation queue, PACS/HL7 interfacing research documented.")),
        ("Theatre &amp; Oncology",
         ("Theatre booking and list management, post-operative note update, "
         "Oncology clinic with chemotherapy regimen tracking, AI treatment summary "
         "(consent-gated).")),
        ("Emergency Access",
         ("Break-glass emergency override with full audit trail, time-limited access tokens, "
         "supervisor alert dispatching, admin audit-trail view.")),
        ("Telemedicine",
         ("WebRTC virtual consultation room with real-time signalling via Socket.IO, "
         "in-call clinical notes, prescription drafting. Feature-flagged "
         "(ENABLE_TELEMEDICINE) pending regulatory sign-off.")),
    ]))
    story.append(Spacer(1, 6 * mm))

    # 3. Administrative Modules
    story.append(Paragraph("3. Administrative &amp; Operational Modules", styles["section_h"]))
    story.append(create_hr())
    story.append(build_feature_table(styles, [
        ("Billing &amp; Finance",
         ("Invoice generation, multi-payment allocation (cash, M-Pesa STK Push, insurance), "
         "unreconciled-charges aggregation, revenue analytics.")),
        ("M-Pesa Integration",
         ("Safaricom Daraja API STK Push and C2B callback handling, payment receipting, "
         "patient-portal self-pay flow.")),
        ("SHA/SHIF Insurance",
         ("InsuranceScheme &amp; PatientInsurance models, claim adjudication workflow, "
         "approval/rejection tracking, analytics breakdown.")),
        ("HR &amp; Credentialing",
         ("Staff roster management, professional-licence upload, expiry-date warning alerts, "
         "StaffCredential model with admin view.")),
        ("Stores &amp; Inventory",
         "Central stores issue/return workflow, stock-level tracking, reorder triggers."),
        ("Mortuary",
         "Deceased-patient registration, body release workflow."),
        ("Analytics Dashboard",
         ("Executive dashboard (Chart.js): bed occupancy trend, 30-day admission curve, "
         "revenue breakdown by payment channel, insurance claim approval ratios. "
         "JSON API for BI tool integration.")),
        ("Patient Self-Service Portal",
         ("Separate PatientUser authentication, appointment booking, lab-result viewing "
         "(gated), billing history, M-Pesa self-pay, profile management with audit log.")),
        ("Outbound Communications",
         ("Flask-Mail email driver, SMS sandbox abstraction, OutboundNotificationLog, "
         "5 event triggers (appointments, labs, billing, payments, claims), "
         "24-hour appointment reminder scheduler.")),
        ("Audit &amp; Logging",
         ("AuditLog DB model, @audited decorator on all write routes, SIEM-export endpoint, "
         "full structured event log with user/IP/timestamp/diff.")),
    ]))
    story.append(Spacer(1, 6 * mm))

    # 4. Interoperability
    story.append(Paragraph("4. Interoperability &amp; Standards", styles["section_h"]))
    story.append(create_hr())
    story.append(build_feature_table(styles, [
        ("FHIR R4 API",
         ("RESTful /api/fhir/R4 endpoints for Patient, Observation, MedicationRequest, "
         "DiagnosticReport resources. Enables integration with national health exchanges.")),
        ("DHIS2 / KHIS Export",
         ("/api/khis blueprint: automated aggregate report generation and push to Kenya's "
         "national DHIS2 instance for MOH reporting compliance.")),
        ("JWT REST API",
         ("Bearer-token authenticated /api/* endpoints for mobile clients, "
         "third-party EHR connectors, and BI dashboards.")),
        ("M-Pesa (Daraja)",
         "STK Push initiation and C2B webhook for real-time payment reconciliation."),
        ("ICD-10 Coding",
         ("50+ curated ICD-10-CM codes across all departments; WHO API credential "
         "requirement documented for full 70,000-code database ingestion.")),
        ("DICOM / PACS",
         ("DICOM Web upload &amp; Cornerstone.js viewer. HL7 MLLP/ASTM LIS interfacing "
         "architecture documented; implementation pending vendor selection.")),
    ]))
    story.append(Spacer(1, 6 * mm))

    # 5. Security & Compliance
    story.append(Paragraph("5. Security &amp; Compliance", styles["section_h"]))
    story.append(create_hr())
    story.append(Paragraph(
        "Security is implemented as a layered defence-in-depth strategy across system layers:",
        styles["body"],
    ))
    story.append(build_feature_table(styles, [
        ("Authentication", ("TOTP-based MFA, 5-attempt account lockout (15-minute timeout), "
         "bcrypt/PBKDF2 password hashing, session signed with SECRET_KEY.")),
        ("Authorisation", ("Role-based access control (RBAC) across 12 roles; "
         "@break_glass_required decorator for emergency overrides with full audit trail.")),
        ("Encryption at Rest", ("Fernet AES-128-CBC + HMAC EncryptedString column type "
         "applied to all patient PII identity fields.")),
        ("Data Protection Act 2019", ("PatientConsent model, Subject Access Request JSON export, "
         "patient anonymisation route, data-residency hard stop documented.")),
        ("CSRF Protection", ("Flask-WTF CSRF tokens on all state-changing forms; "
         "JWT API endpoints explicitly exempted.")),
        ("Rate Limiting", "Flask-Limiter on login (5/min POST), disabled in test mode."),
        ("CI Security Gates", ("pip-audit (17 known advisories documented), bandit SAST, "
         "import-order linting — all enforced in GitHub Actions without bypasses.")),
        ("Session Security", ("Redis-backed sessions, 30-minute idle timeout, SameSite=Lax, "
         "HttpOnly, Secure cookie flags.")),
    ], col_widths=[4.5 * cm, 12.5 * cm]))
    story.append(Spacer(1, 6 * mm))

    # 6. Technical Architecture
    story.append(Paragraph("6. Technical Architecture", styles["section_h"]))
    story.append(create_hr())
    story.append(Paragraph("6.1 Stack Overview", styles["sub_h"]))
    story.append(build_feature_table(styles, [
        ("Backend", "Python 3.12 · Flask 3.x · SQLAlchemy ORM · Flask-Migrate (Alembic)"),
        ("Database", "PostgreSQL 16 (primary) · SQLite (test / offline fallback)"),
        ("Cache / Queue", "Redis 7 (session store + Celery broker) · Celery 5.4 (async tasks)"),
        ("Real-time", "Flask-SocketIO · WebRTC (telemedicine signalling)"),
        ("Auth", "Flask-Login · Flask-JWT-Extended · PyOTP (TOTP MFA)"),
        ("AI / NLP", "NVIDIA NIM API client · local HuggingFace summariser · DrugCentral DDI DB"),
        ("Containers", "Docker + Docker Compose (web, db, redis, celery_worker, nlp services)"),
        ("CI/CD", "GitHub Actions: lint → bandit → pip-audit → pytest (214 tests, 41.6% cov)"),
        ("Offline / PWA", "Service Worker sw.js · Web App Manifest · offline.html fallback"),
    ], col_widths=[4.5 * cm, 12.5 * cm]))

    story.append(Paragraph("6.2 Performance Baseline", styles["sub_h"]))
    story.append(Paragraph(
        "Hot-path benchmarking was executed with a session-authenticated concurrent load "
        "test (C = 20 simultaneous workers). Results recorded in docs/load_test_results.md:",
        styles["body"],
    ))
    story.append(build_feature_table(styles, [
        ("Patient Registration", "0.0% error rate · sub-200 ms p95 at C=20"),
        ("Prescription Sign-off", "0.0% error rate · sub-250 ms p95 at C=20"),
        ("Billing Payment", "0.0% error rate · sub-200 ms p95 at C=20"),
        ("Health Check /healthz", "1,200 req/sec sustained · Flask-Limiter lockout verified"),
    ], col_widths=[5.5 * cm, 11.5 * cm]))
    story.append(Spacer(1, 6 * mm))

    # 7. Audit Readiness
    story.append(Paragraph("7. Audit &amp; Regulatory Readiness", styles["section_h"]))
    story.append(create_hr())
    story.append(Paragraph(
        "Three independent audit-preparation packages are maintained within the repository:",
        styles["body"],
    ))
    for doc_name, desc in [
        ("security_audit_readiness.md",
         ("Maps OWASP Top-10 controls to implementation routes. "
         "Identifies 17 open pip-audit advisories with risk ratings and mitigations.")),
        ("clinical_safety_review_packaging.md",
         ("Documents Clinical Decision Support rules, AI governance framework "
         "(input validation, consent gating, audit timer), and allergy registry.")),
        ("accessibility_audit_report.md",
         ("WCAG 2.1 AA compliance verification covering keyboard navigation, ARIA landmarks, "
         "and screen-reader compatibility.")),
    ]:
        story.append(Paragraph(f"<b>docs/{doc_name}</b>", styles["sub_h"]))
        story.append(Paragraph(desc, styles["body"]))

    story.append(Spacer(1, 6 * mm))

    # 8. Roadmap & Closing
    story.append(Paragraph("8. Recommended Next Steps", styles["section_h"]))
    story.append(create_hr())
    story.append(build_feature_table(styles, [
        ("ICD-10 Full Database", "Expand 50-item curated array to full ICD-10-CM FTS5 index via WHO API."),
        ("KRA eTIMS Middleware", "Implement tax middleware for automated KRA electronic invoicing."),
        ("Controlled Drugs", "Schedule I/II/III dispensing log with dual-signature authorization."),
        ("Hardware Interfacing", "ASTM E1381 MLLP listener for lab analyzers and PACS DICOM worklists."),
        ("Penetration Testing", "Third-party OWASP web application penetration audit prior to go-live."),
    ], col_widths=[5.5 * cm, 11.5 * cm]))

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("9. Contact &amp; Licensing", styles["section_h"]))
    story.append(create_hr())
    story.append(Paragraph(
        "This document is confidential and intended solely for designated stakeholders. "
        "The HMIS codebase is maintained under a proprietary license. "
        "For deployment inquiries, clinical configuration, or integration support, contact "
        "the engineering team via the repository portal.",
        styles["body"],
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(f"Document generated: {datetime.now(timezone.utc).date().strftime('%d %B %Y')}", styles["caption"]))

    return story


# ── Main Entrypoint ───────────────────────────────────────────────────────────
def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "HMIS_Product_Overview.pdf"

    doc = BaseDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=25 * mm,
        bottomMargin=20 * mm,
        title="HMIS — Hospital Management Information System",
        author="HMIS Development Team",
        subject="Product Overview & Capability Reference",
    )

    # Frame & Template Setup
    cover_frame = Frame(
        0, 0, PAGE_WIDTH, PAGE_HEIGHT,
        leftPadding=30 * mm, rightPadding=30 * mm,
        topPadding=0, bottomPadding=30 * mm,
        id="cover_frame",
    )
    inner_frame = Frame(
        20 * mm, 18 * mm,
        PAGE_WIDTH - 40 * mm, PAGE_HEIGHT - 18 * mm - 20 * mm,
        id="inner_frame",
    )

    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[cover_frame]),
        PageTemplate(id="inner", frames=[inner_frame]),
    ])

    styles = make_styles()
    story = build_story(styles)

    # Build PDF with Dynamic Page Counter Canvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅  Professional PDF generated: {out_path}")


if __name__ == "__main__":
    main()
