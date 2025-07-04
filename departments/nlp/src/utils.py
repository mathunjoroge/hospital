import re
import html
import hashlib
from typing import Dict, List, Tuple
from collections import defaultdict
import time
from datetime import datetime
import pytz
import spacy
from textwrap import shorten
from src.database import get_sqlite_connection

from src.nlp import DiseasePredictor
from resources.priority_symptoms import PRIORITY_SYMPTOMS
import logging
from resources.common_fallbacks import fallback_management_plans

logger = logging.getLogger("HIMS-NLP")

# Central timezone configuration
TIME_ZONE = pytz.timezone('Africa/Nairobi')

# Bootstrap styling constants
BOOTSTRAP_CLASSES = {
    "container": "container mt-4",
    "card": "card shadow-sm mb-4",
    "card_header": "card-header bg-primary text-white",
    "badge_primary": "badge bg-primary ms-2",
    "badge_priority": "badge bg-danger ms-2",
    "badge_keyword": "badge bg-info text-dark",
    "badge_cui": "badge bg-secondary",
    "section_heading": "border-bottom pb-2",
}

def _load_management_plans() -> Dict:
    """Lazy load management plans and lab tests for diseases."""
    try:
        with get_sqlite_connection() as conn:
            cursor = conn.cursor()
            
            # Query to fetch management plans
            cursor.execute("""
                SELECT d.name, dmp.plan
                FROM disease_management_plans dmp
                JOIN diseases d ON dmp.disease_id = d.id
            """)
            management_plans = {row['name'].lower(): {'plan': row['plan']} for row in cursor.fetchall()}
            
            # Query to fetch lab tests
            cursor.execute("""
                SELECT d.name, dl.lab_test, dl.description
                FROM disease_labs dl
                JOIN diseases d ON dl.disease_id = d.id
            """)
            lab_tests = cursor.fetchall()
            
            # Combine lab tests with management plans
            for row in lab_tests:
                disease_name = row['name'].lower()
                if disease_name not in management_plans:
                    management_plans[disease_name] = {'plan': '', 'lab_tests': []}
                if 'lab_tests' not in management_plans[disease_name]:
                    management_plans[disease_name]['lab_tests'] = []
                management_plans[disease_name]['lab_tests'].append({
                    'test': row['lab_test'],
                    'description': row['description'] or ''
                })
            
            return management_plans
    except Exception as e:
        logger.error(f"Failed to load management plans and lab tests: {e}")
        return fallback_management_plans

def generate_primary_diagnosis_html(primary_diagnosis: dict) -> str:
    """Generate HTML for primary diagnosis using Bootstrap classes."""
    if not primary_diagnosis:
        return f"""
        <div class="alert alert-danger" role="alert">
            <h5 class="alert-heading">Primary Diagnosis</h5>
            <p>No primary diagnosis identified</p>
        </div>
        """
    
    return f"""
    <div class="alert alert-success" role="alert">
        <h5 class="alert-heading">Primary Diagnosis</h5>
        <p><strong>Disease:</strong> {html.escape(primary_diagnosis.get('disease', 'N/A'))}</p>
        <p><span class="{BOOTSTRAP_CLASSES['badge_primary']}">Score: {primary_diagnosis.get('score', 'N/A')}</span></p>
    </div>
    """

def generate_differential_diagnoses_html(differential_diagnoses: list) -> str:
    """Generate HTML for differential diagnoses using Bootstrap classes."""
    if not differential_diagnoses:
        return ""
    
    table_rows = "".join(
        f'<tr><td class="border p-2">{html.escape(diag["disease"])}</td><td class="border p-2">{diag["score"]}</td></tr>'
        for diag in differential_diagnoses
    )
    return f"""
    <div class="mb-4">
        <h5 class="{BOOTSTRAP_CLASSES['section_heading']}"><i class="bi bi-list-ul"></i> Differential Diagnoses</h5>
        <table class="table table-bordered">
            <thead class="table-light">
                <tr><th>Disease</th><th>Score</th></tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
    """

def generate_entities_html(entities: list) -> str:
    """Generate HTML for extracted entities with collapsible associated diseases using Bootstrap classes."""
    if not entities:
        return ""
    
    entity_items = []
    for entity in entities:
        text, label, context = entity
        is_priority = text.lower() in PRIORITY_SYMPTOMS
        diseases = context.get("associated_diseases", [])
        disease_symptom_map = context.get("disease_symptom_map", {})
        
        # Deduplicate and group similar diseases
        disease_groups = defaultdict(list)
        for disease in diseases:
            base_name = re.sub(r'\s*\(.*?\)|\s*(disorder|syndrome|deficiency|type)\b', '', disease, flags=re.IGNORECASE).strip()
            disease_groups[base_name].append(disease)
        
        # Limit displayed diseases to 5
        disease_items = []
        for i, (base_name, variants) in enumerate(disease_groups.items()):
            if i >= 5:
                break
            all_symptoms = set()
            for variant in variants:
                if variant in disease_symptom_map:
                    all_symptoms.update(disease_symptom_map[variant])
            symptom_list = ", ".join(sorted(all_symptoms)[:3])
            if len(all_symptoms) > 3:
                symptom_list += ", ..."
            disease_items.append(f"<li>{html.escape(variants[0])}: {html.escape(symptom_list)}</li>")
        
        more_count = len(disease_groups) - 5 if len(disease_groups) > 5 else 0
        disease_list = f"""
        <div class="ms-3">
            <strong>Associated Diseases:</strong>
            <ul class="list-group list-group-flush">
                {''.join(disease_items)}
                {'<li><button class="btn btn-link btn-sm show-more-btn" data-entity="{html.escape(text)}" aria-label="Show more diseases">Show More ({more_count})</button></li>' if more_count > 0 else ''}
            </ul>
        </div>
        """
        
        # Generate unique ID for each entity
        entity_id = hashlib.md5(f"{text}{label}".encode()).hexdigest()[:8]
        
        entity_items.append(f"""
        <div class="{BOOTSTRAP_CLASSES['card']} {'border-warning bg-warning-subtle' if is_priority else ''}">
            <div class="card-header d-flex justify-content-between align-items-center">
                <div>
                    <strong>{html.escape(text)}</strong>
                    <span class="{BOOTSTRAP_CLASSES['badge_primary']}">{html.escape(label)}</span>
                    {'<span class="' + BOOTSTRAP_CLASSES['badge_priority'] + '">Priority</span>' if is_priority else ''}
                </div>
                <button class="btn btn-link btn-sm" type="button" data-bs-toggle="collapse" 
                        data-bs-target="#entity-{entity_id}" 
                        aria-expanded="false" 
                        aria-controls="entity-{entity_id}">
                    Toggle Details
                </button>
            </div>
            <div id="entity-{entity_id}" class="collapse">
                <div class="card-body">
                    <p class="small">Severity: {context["severity"]}, Temporal: {html.escape(context["temporal"])}</p>
                    {disease_list if diseases else '<p>No associated diseases</p>'}
                </div>
            </div>
        </div>
        """)
    
    return f"""
    <div class="mb-4">
        <h5 class="{BOOTSTRAP_CLASSES['section_heading']}"><i class="bi bi-list-check"></i> Entities</h5>
        <div>
            {"".join(entity_items)}
        </div>
    </div>
    """

def generate_management_plans_html(management_plans: dict) -> str:
    """Generate HTML for management plans and lab tests using Bootstrap classes."""
    if not management_plans:
        return ""
    
    plans_html = []
    for disease, data in management_plans.items():
        # Escape disease name and plan for HTML safety
        disease_escaped = html.escape(disease)
        plan_escaped = html.escape(data.get('plan', 'No plan available'))
        
        # Generate HTML for lab tests
        lab_tests = data.get('lab_tests', [])
        lab_tests_html = ""
        if lab_tests:
            lab_test_items = "".join(
                f'<li><strong>{html.escape(test["test"])}:</strong> {html.escape(test["description"])}</li>'
                for test in lab_tests
            )
            lab_tests_html = f"""
            <div class="mt-3">
                <h6 class="{BOOTSTRAP_CLASSES['section_heading']}">Recommended Lab Tests</h6>
                <ul class="list-group list-group-flush">
                    {lab_test_items}
                </ul>
            </div>
            """
        
        # Combine plan and lab tests in a single card
        plans_html.append(f"""
        <div class="{BOOTSTRAP_CLASSES['card']}">
            <div class="{BOOTSTRAP_CLASSES['card_header']}">
                <h5 class="mb-0">{disease_escaped}</h5>
            </div>
            <div class="card-body">
                <div><strong>Management Plan:</strong> {plan_escaped}</div>
                {lab_tests_html}
            </div>
        </div>
        """)
    
    return f"""
    <div class="mb-4">
        <h5 class="{BOOTSTRAP_CLASSES['section_heading']}"><i class="bi bi-prescription"></i> Management Plans and Lab Tests</h5>
        <div>
            {"".join(plans_html)}
        </div>
    </div>
    """

def generate_metadata_html(note_id: str, patient_id: str, processed_at: str, processing_time: float) -> str:
    """Generate HTML for metadata section."""
    return f"""
    <div class="row g-3 mb-4">
        <div class="col-12 col-md-4">
            <div class="{BOOTSTRAP_CLASSES['card']}">
                <div class="card-body">
                    <h6 class="card-title text-mutedLiberty">Patient ID</h6>
                    <p class="card-text">{patient_id}</p>
                </div>
            </div>
        </div>
        <div class="col-12 col-md-4">
            <div class="{BOOTSTRAP_CLASSES['card']}">
                <div class="card-body">
                    <h6 class="card-title text-muted">Processed At</h6>
                    <p class="card-text">{processed_at}</p>
                </div>
            </div>
        </div>
        <div class="col-12 col-md-4">
            <div class="{BOOTSTRAP_CLASSES['card']}">
                <div class="card-body">
                    <h6 class="card-title text-muted">Processing Time</h6>
                    <p class="card-text">{processing_time:.3f} seconds</p>
                </div>
            </div>
        </div>
    </div>
    """

def generate_html_response(data: Dict[str, any], status_code: int = 200) -> str:
    """Generate HTML div content for the /process_note endpoint using Bootstrap classes."""
    try:
        if status_code == 404:
            return f"""
            <div class="{BOOTSTRAP_CLASSES['container']}">
                <div class="alert alert-danger" role="alert">
                    <h4 class="alert-heading">Error: Note Not Found</h4>
                    <p>The requested note was not found in the database.</p>
                    <p>Please verify the note ID or contact <a href="mailto:support@example.com">support@example.com</a>.</p>
                </div>
            </div>
            """

        note_id = html.escape(str(data.get("note_id", "Unknown")))
        patient_id = html.escape(data.get("patient_id", "Unknown"))
        primary_diagnosis = data.get("primary_diagnosis", {})
        differential_diagnoses = data.get("differential_diagnoses", [])
        keywords = [html.escape(k) for k in data.get("keywords", [])]
        cuis = [html.escape(c) for c in data.get("cuis", [])]
        entities = data.get("entities", [])
        summary = html.escape(data.get("summary", ""))
        management_plans = data.get("management_plans", {})
        
        created_at = data.get("created_at")
        processed_at = (
            datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            .astimezone(TIME_ZONE)
            .strftime("%Y-%m-%d %H:%M:%S %Z")
            if created_at
            else datetime.now(TIME_ZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
        )
        processed_at = html.escape(processed_at)
        processing_time = data.get("processing_time", 0.0)

        primary_html = generate_primary_diagnosis_html(primary_diagnosis)
        differential_html = generate_differential_diagnoses_html(differential_diagnoses)
        entities_html = generate_entities_html(entities)
        management_html = generate_management_plans_html(management_plans)
        metadata_html = generate_metadata_html(note_id, patient_id, processed_at, processing_time)

        summary = shorten(summary, width=500, placeholder="...") if summary else ""

        return f"""
        <div class="{BOOTSTRAP_CLASSES['container']}">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <div class="{BOOTSTRAP_CLASSES['card']}">
                <div class="{BOOTSTRAP_CLASSES['card_header']}">
                    <h3 class="mb-0"><i class="bi bi-clipboard-pulse"></i> AI Clinical Analysis - Note ID {note_id}</h3>
                </div>
                <div class="card-body">
                    <div class="mb-4">
                        <h5 class="{BOOTSTRAP_CLASSES['section_heading']}">Search Report</h5>
                        <input type="text" class="form-control" id="reportSearch" placeholder="Search keywords, entities, or diagnoses..." onkeyup="filterReport()" aria-label="Search report content">
                    </div>
                    
                    {metadata_html}
                    {primary_html}
                    {differential_html}

                    <div class="mb-4">
                        <h5 class="{BOOTSTRAP_CLASSES['section_heading']}"><i class="bi bi-tags"></i> Keywords</h5>
                        <div class="d-flex flex-wrap gap-2">
                            {"".join(f'<span class="{BOOTSTRAP_CLASSES["badge_keyword"]}">{keyword}</span>' for keyword in keywords)}
                        </div>
                    </div>

                    <div class="mb-4">
                        <h5 class="{BOOTSTRAP_CLASSES['section_heading']}"><i class="bi bi-code"></i> CUIs</h5>
                        <div class="d-flex flex-wrap gap-2">
                            {"".join(f'<span class="{BOOTSTRAP_CLASSES["badge_cui"]}" data-bs-toggle="tooltip" title="Unified Medical Language System CUI">{cui}</span>' for cui in cuis)}
                        </div>
                    </div>

                    {entities_html}
                    {management_html}

                    <div class="{BOOTSTRAP_CLASSES['card']}">
                        <div class="card-body">
                            <h5 class="card-title {BOOTSTRAP_CLASSES['section_heading']}"><i class="bi bi-file-text"></i> Summary</h5>
                            <p class="card-text">{summary or 'No summary available.'}</p>
                        </div>
                    </div>

                    <div class="d-flex gap-2">
                        <button class="btn btn-primary" onclick="window.print()" aria-label="Download report as PDF"><i class="bi bi-printer"></i> Download as PDF</button>
                        <button class="btn btn-outline-secondary" onclick="copyToClipboard()" aria-label="Copy report to clipboard">Copy Report</button>
                    </div>
                </div>
            </div>
        </div>
        <script>
            function filterReport() {{
                const input = document.getElementById('reportSearch').value.toLowerCase();
                const cards = document.querySelectorAll('.card, .alert, .badge');
                cards.forEach(card => {{
                    const text = card.innerText.toLowerCase();
                    card.style.display = text.includes(input) ? '' : 'none';
                }});
            }}
            function copyToClipboard() {{
                const content = document.querySelector('.{BOOTSTRAP_CLASSES['container'].split()[0]}').innerText;
                navigator.clipboard.writeText(content).then(() => {{
                    alert('Report copied to clipboard!');
                }});
            }}
        </script>
        """
    except Exception as e:
        logger.error(f"HTML generation error: {e}")
        return f"""
        <div class="{BOOTSTRAP_CLASSES['container']}">
            <div class="alert alert-danger" role="alert">
                <h4 class="alert-heading">Error Generating Report</h4>
                <p>An unexpected error occurred: {html.escape(str(e))}</p>
                <p>Please try again or contact <a href="mailto:support@example.com">support@example.com</a>.</p>
            </div>
        </div>
        """

def prepare_note_for_nlp(soap_note: Dict) -> str:
    """Prepare a SOAP note for NLP processing."""
    fields = [
        soap_note.get('situation', ''),
        soap_note.get('hpi', ''),
        soap_note.get('symptoms', ''),
        soap_note.get('aggravating_factors', ''),
        soap_note.get('alleviating_factors', ''),
        soap_note.get('medical_history', ''),
        soap_note.get('medication_history', ''),
        soap_note.get('assessment', ''),
        soap_note.get('recommendation', ''),
        soap_note.get('additional_notes', ''),
        soap_note.get('ai_notes', '')
    ]
    return ' '.join(f for f in fields if f).strip()

def humanize_list(items: List[str]) -> str:
    """Format a list of items in a natural language string."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"

def generate_summary(text: str, soap_note: Dict, max_sentences: int = 4, include_fields: List[str] = None, doc=None) -> str:
    """Generate a concise and natural clinical summary, incorporating key SOAP note fields."""
    include_fields = include_fields or ['symptoms', 'findings', 'recommendations']
    if not text:
        return ""
    
    start_time = time.time()
    if not doc:
        doc = DiseasePredictor.nlp(text)
    
    symptoms = set()
    findings = []
    recommendations = []
    temporal_info = ""
    temporal_pattern = re.compile(
        r"\b(\d+\s*(day|days|week|weeks|month|months|year|years)\s*(ago)?)\b", 
        re.IGNORECASE
    )
    
    text_lower = text.lower()
    for sent in doc.sents:
        sent_text = sent.text.strip()
        match = temporal_pattern.search(sent_text)
        if match and not temporal_info:
            temporal_info = match.group(0)
        if 'symptoms' in include_fields:
            for term in DiseasePredictor.clinical_terms:
                if term in sent_text.lower():
                    symptoms.add(term)
    
    if 'findings' in include_fields and soap_note.get('assessment'):
        for term in DiseasePredictor.clinical_terms:
            if term in soap_note['assessment'].lower():
                findings.append(term)
    
    if 'recommendations' in include_fields and soap_note.get('recommendation'):
        recommendations.append(soap_note['recommendation'])
    
    summary_parts = []
    if symptoms and 'symptoms' in include_fields:
        symptoms_str = humanize_list(sorted(symptoms))
        presentation = f"The patient presents with {symptoms_str}"
        if temporal_info:
            presentation += f", starting {temporal_info}"
        summary_parts.append(presentation + ".")
    
    for finding in findings:
        if 'findings' in include_fields:
            summary_parts.append(f"Findings include {finding}.")
    
    for rec in recommendations:
        if 'recommendations' in include_fields:
            summary_parts.append(f"Recommended: {rec}.")
    
    summary = " ".join(summary_parts[:max_sentences]).strip()
    summary = re.sub(r"\s+", " ", summary)
    summary = re.sub(r"\.+", ".", summary)
    
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    logger.debug(f"Summary generation took {time.time() - start_time:.3f} seconds")
    logger.debug(f"Generated summary: {summary}")
    
    return summary if summary else text[:150] + "..."