import re
import html
import hashlib
import time
import logging
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime
from textwrap import shorten

import pytz
from src.database import get_sqlite_connection
from src.nlp import DiseasePredictor
from resources.priority_symptoms import PRIORITY_SYMPTOMS
from resources.common_fallbacks import fallback_management_plans
# Assuming cancer plans are moved to a separate resource file for better organization
# from resources.cancer_management_plans import cancer_management_plans

logger = logging.getLogger("HIMS-NLP")

# --- CONFIGURATION & SETUP ---

# Central timezone configuration
TIME_ZONE = pytz.timezone('Africa/Nairobi')

# Define terms to ignore during summary entity extraction to avoid noise.
SUMMARY_INVALID_TERMS = {'mg', 'ms', 'g', 'ml', 'mm', 'ng', 'dl', 'hr'}

# Bootstrap styling constants
BOOTSTRAP_CLASSES = {
    "container": "container mt-4",
    "card": "card shadow-sm mb-4",
    "card_header": "card-header bg-primary text-white",
    "card_body": "card-body",
    "alert_success": "alert alert-success",
    "alert_danger": "alert alert-danger",
    "badge_primary": "badge bg-primary ms-2",
    "badge_priority": "badge bg-danger ms-2",
    "badge_keyword": "badge bg-info text-dark",
    "badge_cui": "badge bg-secondary",
    "badge_warning": "badge bg-warning text-dark",
    "section_heading": "border-bottom pb-2 mb-3",
}

# --- DATA LOADING ---

def load_management_plans() -> Dict[str, Dict[str, Any]]:
    """Load management plans and lab tests for diseases."""
    try:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            management_plans = defaultdict(lambda: {'plan': '', 'lab_tests': []})

            # 1. Fetch general management plans
            cursor.execute("""
                SELECT d.name, dmp.plan
                FROM disease_management_plans dmp
                JOIN diseases d ON dmp.disease_id = d.id
            """)
            for row in cursor.fetchall():
                management_plans[row['name'].lower()]['plan'] = row['plan']

            # 2. Fetch and associate lab tests
            cursor.execute("""
                SELECT d.name, dl.lab_test, dl.description
                FROM disease_labs dl
                JOIN diseases d ON dl.disease_id = d.id
            """)
            for row in cursor.fetchall():
                management_plans[row['name'].lower()]['lab_tests'].append({
                    'test': row['lab_test'],
                    'description': row['description'] or ''
                })

            # 3. Convert defaultdict to dict and integrate cancer-specific plans
            # (It's better practice to move these to a separate `resources` file)
            cancer_plans = {
                'prostate cancer': {'plan': 'Refer to oncologist; order prostate biopsy', 'lab_tests': [{'test': 'PSA follow-up', 'description': 'Monitor PSA levels in 4-6 weeks'}]},
                'lymphoma': {'plan': 'Order lymph node biopsy; consider PET scan', 'lab_tests': [{'test': 'LDH', 'description': 'Assess lymphoma activity'}]},
                # ... other cancer plans ...
            }
            final_plans = dict(management_plans)
            final_plans.update(cancer_plans)

            return final_plans
    except Exception as e:
        logger.error(f"Failed to load management plans: {e}", exc_info=True)
        return fallback_management_plans

# --- TEXT & NLP UTILITIES ---

def prepare_note_for_nlp(soap_note: Dict) -> str:
    """Prepare a SOAP note for NLP processing."""
    fields = [
        soap_note.get('situation', ''), soap_note.get('hpi', ''),
        soap_note.get('symptoms', ''), soap_note.get('aggravating_factors', ''),
        soap_note.get('alleviating_factors', ''), soap_note.get('medical_history', ''),
        soap_note.get('medication_history', ''), soap_note.get('assessment', ''),
        soap_note.get('recommendation', ''), soap_note.get('additional_notes', ''),
        soap_note.get('ai_notes', '')
    ]
    return ' '.join(filter(None, fields)).strip()

def humanize_list(items: List[str]) -> str:
    """Format a list of items into a natural language string."""
    if not items: return ""
    if len(items) == 1: return items[0]
    if len(items) == 2: return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"

def generate_summary(text: str, soap_note: Dict, max_sentences: int = 4, doc=None) -> str:
    """Generate a concise and natural clinical summary."""
    if not text.strip():
        return "No summary available."

    start_time = time.time()
    if not doc:
        doc = DiseasePredictor.nlp(text)

    # Extract key info from text
    symptoms = sorted({
        term for sent in doc.sents for term in DiseasePredictor.clinical_terms
        if term in sent.text.lower() and term not in SUMMARY_INVALID_TERMS
    })
    temporal_match = re.search(r"\b(\d+\s*(days?|weeks?|months?|years?)\s*(ago)?)\b", text, re.IGNORECASE)
    temporal_info = f", which began approximately {temporal_match.group(0)}" if temporal_match else ""

    # Extract structured data
    lab_abnormalities = [lab for lab in soap_note.get('lab_abnormalities', []) if lab.lower() not in SUMMARY_INVALID_TERMS]
    recommendation = soap_note.get('recommendation', '').strip()

    summary_parts = []
    if symptoms:
        summary_parts.append(f"The patient presents with {humanize_list(symptoms)}{temporal_info}.")
    if lab_abnormalities:
        summary_parts.append(f"Noteworthy lab abnormalities include {humanize_list(lab_abnormalities)}.")
    if recommendation:
        summary_parts.append(f"The current recommendation is: {recommendation}")
    
    summary = " ".join(summary_parts[:max_sentences]).strip()
    logger.debug(f"Summary generation took {time.time() - start_time:.3f} seconds.")

    if not summary:
        return shorten(text, width=250, placeholder="...")
    
    return re.sub(r'\s+', ' ', summary)

# --- HTML REPORT GENERATION ---

def _generate_component_html(title: str, icon_class: str, content: str, is_visible: bool = True) -> str:
    """A helper to generate a standard, filterable HTML section card."""
    if not is_visible or not content:
        return ""
    return f"""
    <div class="mb-4 filterable-section">
        <h5 class="{BOOTSTRAP_CLASSES['section_heading']}"><i class="bi {icon_class}"></i> {title}</h5>
        <div>{content}</div>
    </div>
    """

def generate_primary_diagnosis_html(primary_diagnosis: dict) -> str:
    """Generate HTML for primary diagnosis."""
    if not primary_diagnosis:
        return f"""
        <div class="{BOOTSTRAP_CLASSES['alert_danger']}" role="alert">
            <h5 class="alert-heading">Primary Diagnosis</h5>
            <p>No primary diagnosis identified.</p>
        </div>
        """
    disease = html.escape(primary_diagnosis.get('disease', 'N/A'))
    score = primary_diagnosis.get('score', 0.0)
    return f"""
    <div class="{BOOTSTRAP_CLASSES['alert_success']}" role="alert">
        <h5 class="alert-heading">Primary Diagnosis</h5>
        <p class="mb-1"><strong>Disease:</strong> {disease}</p>
        <p class="mb-0"><span class="{BOOTSTRAP_CLASSES['badge_primary']}">Confidence Score: {score:.2f}</span></p>
    </div>
    """

def generate_differential_diagnoses_html(diagnoses: list) -> str:
    """Generate HTML for differential diagnoses."""
    if not diagnoses:
        return ""
    rows = "".join(
        f'<tr><td>{html.escape(diag["disease"])}</td><td>{diag["score"]:.2f}</td></tr>'
        for diag in diagnoses
    )
    return f"""
    <table class="table table-bordered table-hover">
        <thead class="table-light">
            <tr><th scope="col">Disease</th><th scope="col">Score</th></tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """

def generate_entities_html(entities: list) -> str:
    """Generate HTML for extracted entities."""
    if not entities:
        return ""
    
    entity_cards = []
    for text, label, context in entities:
        entity_id = hashlib.md5(f"{text}{label}".encode()).hexdigest()[:8]
        is_priority = text.lower() in PRIORITY_SYMPTOMS
        priority_badge = f'<span class="{BOOTSTRAP_CLASSES["badge_priority"]}">Priority</span>' if is_priority else ''
        card_border_class = 'border-danger' if is_priority else ''
        
        # ✅ FIX APPLIED HERE: Converted severity and temporal values to strings before escaping.
        severity_str = html.escape(str(context.get("severity", "N/A")))
        temporal_str = html.escape(str(context.get("temporal", "N/A")))

        entity_cards.append(f"""
        <div class="{BOOTSTRAP_CLASSES['card']} {card_border_class} mb-2">
            <div class="card-header d-flex justify-content-between align-items-center p-2">
                <div>
                    <strong>{html.escape(text)}</strong>
                    <span class="{BOOTSTRAP_CLASSES['badge_primary']}">{html.escape(label)}</span>
                    {priority_badge}
                </div>
                <button class="btn btn-sm btn-outline-secondary" type="button" data-bs-toggle="collapse" 
                        data-bs-target="#entity-{entity_id}" aria-expanded="false">
                    Details
                </button>
            </div>
            <div id="entity-{entity_id}" class="collapse">
                <div class="{BOOTSTRAP_CLASSES['card_body']} small p-2">
                    <p class="mb-0"><strong>Severity:</strong> {severity_str} | <strong>Temporal:</strong> {temporal_str}</p>
                </div>
            </div>
        </div>
        """)
    return "".join(entity_cards)

def generate_management_plans_html(management_plans: dict) -> str:
    """Generate HTML for management plans and lab tests."""
    if not management_plans:
        return ""
    plan_cards = []
    for disease, data in management_plans.items():
        plan_escaped = html.escape(data.get('plan', 'No plan available.'))
        lab_tests_html = ""
        if lab_tests := data.get('lab_tests'):
            items = "".join(
                f'<li class="list-group-item"><strong>{html.escape(t["test"])}:</strong> {html.escape(t["description"])}</li>'
                for t in lab_tests
            )
            lab_tests_html = f'<h6>Recommended Lab Tests</h6><ul class="list-group list-group-flush">{items}</ul>'
        
        plan_cards.append(f"""
        <div class="{BOOTSTRAP_CLASSES['card']}">
            <div class="{BOOTSTRAP_CLASSES['card_header']}">
                <h6 class="mb-0 text-capitalize">{html.escape(disease)}</h6>
            </div>
            <div class="{BOOTSTRAP_CLASSES['card_body']}">
                <p><strong>Management Plan:</strong> {plan_escaped}</p>
                {lab_tests_html}
            </div>
        </div>
        """)
    return "".join(plan_cards)

def generate_metadata_html(note_id: str, patient_id: str, processed_at: str, processing_time: float) -> str:
    """Generate HTML for metadata section."""
    card_class = BOOTSTRAP_CLASSES['card']
    body_class = BOOTSTRAP_CLASSES['card_body']
    return f"""
    <div class="row g-3 mb-4">
        <div class="col-12 col-md-4"><div class="{card_class}"><div class="{body_class}">
            <h6 class="card-title text-muted">Patient ID</h6><p class="card-text mb-0">{patient_id}</p>
        </div></div></div>
        <div class="col-12 col-md-4"><div class="{card_class}"><div class="{body_class}">
            <h6 class="card-title text-muted">Processed At</h6><p class="card-text mb-0">{processed_at}</p>
        </div></div></div>
        <div class="col-12 col-md-4"><div class="{card_class}"><div class="{body_class}">
            <h6 class="card-title text-muted">Processing Time</h6><p class="card-text mb-0">{processing_time:.3f} seconds</p>
        </div></div></div>
    </div>
    """
    
def _get_javascript_html() -> str:
    """Returns the inline JavaScript for report interactivity."""
    return """
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            let searchTimeout;
            const searchInput = document.getElementById('reportSearch');
            if (searchInput) {
                searchInput.addEventListener('keyup', () => {
                    clearTimeout(searchTimeout);
                    searchTimeout = setTimeout(() => {
                        const query = searchInput.value.toLowerCase();
                        document.querySelectorAll('.filterable-section').forEach(section => {
                            const content = section.innerText.toLowerCase();
                            section.style.display = content.includes(query) ? '' : 'none';
                        });
                    }, 300);
                });
            }
            document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => new bootstrap.Tooltip(el));
        });

        function showToast(message, type = 'success') {
            const toastEl = document.createElement('div');
            toastEl.className = `toast align-items-center text-bg-${type} border-0 position-fixed bottom-0 end-0 m-3`;
            toastEl.innerHTML = `<div class="d-flex"><div class="toast-body">${message}</div><button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>`;
            document.body.appendChild(toastEl);
            const toast = new bootstrap.Toast(toastEl, { delay: 4000 });
            toast.show();
            toastEl.addEventListener('hidden.bs.toast', () => toastEl.remove());
        }

        async function copyToClipboard() {
            const reportContent = document.getElementById('main-report-card-body').innerText;
            try {
                await navigator.clipboard.writeText(reportContent);
                showToast('Report copied to clipboard!');
            } catch (err) {
                const textArea = document.createElement("textarea");
                textArea.value = reportContent;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                textArea.remove();
                showToast('Report copied to clipboard (fallback)!');
            }
        }
    </script>
    """

def generate_html_response(data: Dict, status_code: int = 200) -> str:
    """Generate the complete HTML response for the processed note."""
    bc = BOOTSTRAP_CLASSES
    try:
        if status_code != 200:
            error_msg = data.get("error", "The requested note was not found.")
            return f'<div class="{bc["container"]}"><div class="{bc["alert_danger"]}">Error: {html.escape(error_msg)}</div></div>'

        # Safely extract and format data
        note_id = html.escape(str(data.get("note_id", "Unknown")))
        patient_id = html.escape(str(data.get("patient_id", "Unknown")))
        created_at = data.get("created_at")
        processed_at = "N/A"
        if created_at:
            utc_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            processed_at = utc_time.astimezone(TIME_ZONE).strftime("%Y-%m-%d %H:%M:%S %Z")

        # Generate each HTML component
        metadata_html = generate_metadata_html(note_id, patient_id, processed_at, data.get("processing_time", 0.0))
        primary_dx_html = generate_primary_diagnosis_html(data.get("primary_diagnosis", {}))
        
        diff_dx_html = _generate_component_html("Differential Diagnoses", "bi-list-ul",
            generate_differential_diagnoses_html(data.get("differential_diagnoses", [])),
            is_visible=bool(data.get("differential_diagnoses")))

        entities_html = _generate_component_html("Extracted Clinical Entities", "bi-card-list",
            generate_entities_html(data.get("entities", [])),
            is_visible=bool(data.get("entities")))
            
        management_html = _generate_component_html("Management Plans", "bi-prescription",
            generate_management_plans_html(data.get("management_plans", {})),
            is_visible=bool(data.get("management_plans")))
            
        summary_html = _generate_component_html("Clinical Summary", "bi-file-text-fill",
            f"<p>{html.escape(data.get('summary', ''))}</p>",
            is_visible=bool(data.get('summary')))

        # Assemble the final HTML page
        return f"""
        <div class="{bc['container']}">
            <div class="{bc['card']}">
                <div class="{bc['card_header']}">
                    <h3 class="mb-0"><i class="bi bi-clipboard-pulse"></i> AI Clinical Analysis - Note ID {note_id}</h3>
                </div>
                <div class="{bc['card_body']}" id="main-report-card-body">
                    <div class="mb-4">
                        <input type="search" class="form-control" id="reportSearch" placeholder="🔍 Search report content..." aria-label="Search report">
                    </div>
                    <div class="filterable-section">{metadata_html}</div>
                    <div class="filterable-section">{primary_dx_html}</div>
                    {diff_dx_html}
                    {entities_html}
                    {management_html}
                    {summary_html}
                    <div class="d-flex gap-2 mt-4">
                        <button class="btn btn-primary" onclick="window.print()"><i class="bi bi-printer"></i> Print / Save as PDF</button>
                        <button class="btn btn-outline-secondary" onclick="copyToClipboard()"><i class="bi bi-clipboard"></i> Copy Report</button>
                    </div>
                </div>
            </div>
        </div>
        {_get_javascript_html()}
        """
    except Exception as e:
        logger.error(f"Fatal error during HTML generation for Note ID {data.get('note_id', 'N/A')}: {e}", exc_info=True)
        return f'<div class="{bc["container"]}"><div class="{bc["alert_danger"]}">An unexpected error occurred while generating the report.</div></div>'