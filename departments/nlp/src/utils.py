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
    "badge_warning": "badge bg-warning text-dark",
    "section_heading": "border-bottom pb-2",
}

def load_management_plans() -> Dict:
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
            
            # Add cancer-specific management plans
            cancer_plans = {
                'prostate cancer': {
                    'plan': 'Refer to oncologist; order prostate biopsy',
                    'lab_tests': [{'test': 'PSA follow-up', 'description': 'Monitor PSA levels in 4-6 weeks'}],
                    'cancer_follow_up': 'Consider imaging (e.g., CT/MRI) and biopsy if indicated.'
                },
                'lymphoma': {
                    'plan': 'Order lymph node biopsy; consider PET scan',
                    'lab_tests': [{'test': 'LDH', 'description': 'Assess lymphoma activity'}],
                    'cancer_follow_up': 'Monitor symptoms; consider tumor marker tests.'
                },
                'leukemia': {
                    'plan': 'Refer to hematologist; order bone marrow biopsy',
                    'lab_tests': [{'test': 'CBC follow-up', 'description': 'Monitor WBC and other blood counts'}],
                    'cancer_follow_up': 'Consider cytogenetic testing.'
                },
                'lung cancer': {
                    'plan': 'Refer to oncologist; order chest CT and biopsy',
                    'lab_tests': [{'test': 'Sputum cytology', 'description': 'Assess for malignant cells'}],
                    'cancer_follow_up': 'Consider PET scan for staging.'
                },
                'colorectal cancer': {
                    'plan': 'Refer to oncologist; order colonoscopy and biopsy',
                    'lab_tests': [{'test': 'CEA', 'description': 'Monitor colorectal cancer markers'}],
                    'cancer_follow_up': 'Consider CT abdomen/pelvis.'
                },
                'ovarian cancer': {
                    'plan': 'Refer to oncologist; order pelvic ultrasound and biopsy',
                    'lab_tests': [{'test': 'CA-125', 'description': 'Monitor ovarian cancer markers'}],
                    'cancer_follow_up': 'Consider CT/MRI for staging.'
                },
                'pancreatic cancer': {
                    'plan': 'Refer to oncologist; order abdominal CT and biopsy',
                    'lab_tests': [{'test': 'CA 19-9', 'description': 'Monitor pancreatic cancer markers'}],
                    'cancer_follow_up': 'Consider endoscopic ultrasound.'
                },
                'liver cancer': {
                    'plan': 'Refer to oncologist; order liver ultrasound and biopsy',
                    'lab_tests': [{'test': 'AFP', 'description': 'Monitor liver cancer markers'}],
                    'cancer_follow_up': 'Consider MRI liver.'
                },
                'breast cancer': {
                    'plan': 'Refer to oncologist; order mammogram and biopsy',
                    'lab_tests': [{'test': 'BRCA testing', 'description': 'Assess genetic risk'}],
                    'cancer_follow_up': 'Consider breast MRI.'
                }
            }
            management_plans.update(cancer_plans)
            
            return management_plans
    except Exception as e:
        logger.error(f"Failed to load management plans and lab tests: {e}", exc_info=True)
        return fallback_management_plans

def generate_primary_diagnosis_html(primary_diagnosis: dict) -> str:
    """Generate HTML for primary diagnosis using Bootstrap classes."""
    if not primary_diagnosis:
        return """
        <div class="alert alert-danger" role="alert">
            <h5 class="alert-heading">Primary Diagnosis</h5>
            <p>No primary diagnosis identified</p>
        </div>
        """
    
    return """
    <div class="alert alert-success" role="alert">
        <h5 class="alert-heading">Primary Diagnosis</h5>
        <p><strong>Disease:</strong> {}</p>
        <p><span class="{}">Score: {:.2f}</span></p>
    </div>""".format(
        html.escape(primary_diagnosis.get('disease', 'N/A')),
        BOOTSTRAP_CLASSES['badge_primary'],
        primary_diagnosis.get('score', 0.0)
    )

def generate_differential_diagnoses_html(differential_diagnoses: list) -> str:
    """Generate HTML for differential diagnoses using Bootstrap classes."""
    if not differential_diagnoses:
        return ""
    
    table_rows = "".join(
        '<tr><td class="border p-2">{}</td><td class="border p-2">{:.2f}</td></tr>'.format(
            html.escape(diag["disease"]),
            diag["score"]
        )
        for diag in differential_diagnoses
    )
    return """
    <div class="mb-4">
        <h5 class="{}"><i class="bi bi-list-ul"></i> Differential Diagnoses</h5>
        <table class="table table-bordered">
            <thead class="table-light">
                <tr><th scope="col">Disease</th><th scope="col">Score</th></tr>
            </thead>
            <tbody>
                {}
            </tbody>
        </table>
    </div>""".format(
        BOOTSTRAP_CLASSES['section_heading'],
        table_rows
    )

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
            disease_items.append("<li>{}: {}</li>".format(
                html.escape(variants[0]),
                html.escape(symptom_list)
            ))
        
        more_count = len(disease_groups) - 5 if len(disease_groups) > 5 else 0
        disease_list = """
        <div class="ms-3">
            <strong>Associated Diseases:</strong>
            <ul class="list-group list-group-flush">
                {}
                {} 
            </ul>
        </div>""".format(
            ''.join(disease_items) if disease_items else '<li>No associated diseases available</li>',
            '<li><button class="btn btn-link btn-sm show-more-btn" data-entity="{}" aria-label="Show more diseases">Show More ({})</button></li>'.format(
                html.escape(text), more_count
            ) if more_count > 0 else ''
        ) if diseases else '<p>No associated diseases available</p>'
        
        # Generate unique ID for each entity
        entity_id = hashlib.md5(f"{text}{label}".encode()).hexdigest()[:8]
        
        # Include cancer relevance and lab details
        extra_info = []
        if context.get('cancer_relevance'):
            extra_info.append("Cancer Relevance: {:.2f}".format(context['cancer_relevance']))
        if context.get('value') and context.get('unit'):
            extra_info.append("Value: {} {}".format(context['value'], context['unit']))
            if context.get('abnormal'):
                extra_info.append("Abnormal: {}".format(context['abnormal']))
                if context.get('potential_cancer'):
                    extra_info.append("Potential Cancer: {}".format(html.escape(context['potential_cancer'])))
        
        extra_info_html = '<p class="small">' + '; '.join(extra_info) + '</p>' if extra_info else ''
        
        entity_items.append("""
        <div class="{} {}">
            <div class="card-header d-flex justify-content-between align-items-center">
                <div>
                    <strong>{}</strong>
                    <span class="{}">{}</span>
                    {}
                </div>
                <button class="btn btn-link btn-sm" type="button" data-bs-toggle="collapse" 
                        data-bs-target="#entity-{}" 
                        aria-expanded="false" 
                        aria-controls="entity-{}">
                    Toggle Details
                </button>
            </div>
            <div id="entity-{}" class="collapse">
                <div class="card-body">
                    <p class="small">Severity: {}, Temporal: {}</p>
                    {}
                    {}
                </div>
            </div>
        </div>""".format(
            BOOTSTRAP_CLASSES['card'],
            'border-warning bg-warning-subtle' if is_priority else '',
            html.escape(text),
            BOOTSTRAP_CLASSES['badge_primary'],
            html.escape(label),
            '<span class="{}">Priority</span>'.format(BOOTSTRAP_CLASSES['badge_priority']) if is_priority else '',
            entity_id, entity_id, entity_id,
            context["severity"], html.escape(context["temporal"]),
            extra_info_html,
            disease_list
        ))
    
    return """
    <div class="mb-4">
        <h5 class="{}"><i class="bi bi-list-check"></i> Entities</h5>
        <div>
            {}
        </div>
    </div>""".format(
        BOOTSTRAP_CLASSES['section_heading'],
        "".join(entity_items)
    )

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
                '<li><strong>{}:</strong> {}</li>'.format(
                    html.escape(test["test"]),
                    html.escape(test["description"])
                )
                for test in lab_tests
            )
            lab_tests_html = """
            <div class="mt-3">
                <h6 class="{}">Recommended Lab Tests</h6>
                <ul class="list-group list-group-flush">
                    {}
                </ul>
            </div>""".format(
                BOOTSTRAP_CLASSES['section_heading'],
                lab_test_items
            )
        
        # Generate HTML for cancer follow-up
        cancer_follow_up = data.get('cancer_follow_up', '')
        cancer_follow_up_html = """
        <div class="mt-3">
            <h6 class="{}">Cancer Follow-Up</h6>
            <p>{}</p>
        </div>""".format(
            BOOTSTRAP_CLASSES['section_heading'],
            html.escape(cancer_follow_up)
        ) if cancer_follow_up else ""
        
        # Generate HTML for lab follow-up
        lab_follow_up = data.get('lab_follow_up', '')
        lab_follow_up_html = """
        <div class="mt-3">
            <h6 class="{}">Lab Follow-Up</h6>
            <p>{}</p>
        </div>""".format(
            BOOTSTRAP_CLASSES['section_heading'],
            html.escape(lab_follow_up)
        ) if lab_follow_up else ""
        
        # Combine plan and follow-ups in a single card
        plans_html.append("""
        <div class="{}">
            <div class="{}">
                <h5 class="mb-0">{}</h5>
            </div>
            <div class="card-body">
                <div><strong>Management Plan:</strong> {}</div>
                {}
                {}
                {}
            </div>
        </div>""".format(
            BOOTSTRAP_CLASSES['card'],
            BOOTSTRAP_CLASSES['card_header'],
            disease_escaped,
            plan_escaped,
            lab_tests_html,
            cancer_follow_up_html,
            lab_follow_up_html
        ))
    
    return """
    <div class="mb-4">
        <h5 class="{}"><i class="bi bi-prescription"></i> Management Plans and Lab Tests</h5>
        <div>
            {}
        </div>
    </div>""".format(
        BOOTSTRAP_CLASSES['section_heading'],
        "".join(plans_html)
    )

def generate_metadata_html(note_id: str, patient_id: str, processed_at: str, processing_time: float) -> str:
    """Generate HTML for metadata section."""
    return """
    <div class="row g-3 mb-4">
        <div class="col-12 col-md-4">
            <div class="{}">
                <div class="card-body">
                    <h6 class="card-title text-muted">Patient ID</h6>
                    <p class="card-text">{}</p>
                </div>
            </div>
        </div>
        <div class="col-12 col-md-4">
            <div class="{}">
                <div class="card-body">
                    <h6 class="card-title text-muted">Processed At</h6>
                    <p class="card-text">{}</p>
                </div>
            </div>
        </div>
        <div class="col-12 col-md-4">
            <div class="{}">
                <div class="card-body">
                    <h6 class="card-title text-muted">Processing Time</h6>
                    <p class="card-text">{:.3f} seconds</p>
                </div>
            </div>
        </div>
    </div>""".format(
        BOOTSTRAP_CLASSES['card'], patient_id,
        BOOTSTRAP_CLASSES['card'], processed_at,
        BOOTSTRAP_CLASSES['card'], processing_time
    )

def generate_html_response(data: Dict, status_code: int = 200) -> str:
    """Generate HTML div content for the /process_note endpoint using Bootstrap classes."""
    try:
        if status_code == 404:
            return """
            <div class="{}">
                <div class="alert alert-danger" role="alert">
                    <h4 class="alert-heading">Error: Note Not Found</h4>
                    <p>The requested note was not found in the database.</p>
                    <p>Please verify the note ID or contact <a href="mailto:support@example.com">support@example.com</a>.</p>
                </div>
            </div>""".format(BOOTSTRAP_CLASSES['container'])

        note_id = html.escape(str(data.get("note_id", "Unknown")))
        patient_id = html.escape(data.get("patient_id", "Unknown"))
        primary_diagnosis = data.get("primary_diagnosis", {})
        differential_diagnoses = data.get("differential_diagnoses", [])
        keywords = [html.escape(k) for k in data.get("keywords", [])]
        cuis = [html.escape(c) for c in data.get("cuis", [])]
        entities = data.get("entities", [])
        summary = html.escape(data.get("summary", ""))
        management_plans = data.get("management_plans", {})
        lab_abnormalities = [html.escape(l) for l in data.get("lab_abnormalities", [])]
        cancer_risk_score = data.get("cancer_risk_score", 0.0)
        image_cancer_risk = data.get("image_cancer_risk", 0.0)
        
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
        
        lab_abnormalities_html = """
        <div class="mb-4">
            <h5 class="{}"><i class="bi bi-vial"></i> Lab Abnormalities</h5>
            <div class="d-flex flex-wrap gap-2">
                {}
            </div>
        </div>""".format(
            BOOTSTRAP_CLASSES['section_heading'],
            "".join('<span class="{}">{}</span>'.format(BOOTSTRAP_CLASSES['badge_warning'], lab) for lab in lab_abnormalities) or 'None'
        ) if lab_abnormalities else ""
        
        risk_scores_html = """
        <div class="mb-4">
            <h5 class="{}"><i class="bi bi-graph-up"></i> Risk Scores</h5>
            <p><strong>Cancer Risk Score:</strong> {:.2%}</p>
            <p><strong>Image Cancer Risk:</strong> {:.2%}</p>
        </div>""".format(
            BOOTSTRAP_CLASSES['section_heading'],
            cancer_risk_score,
            image_cancer_risk
        ) if cancer_risk_score > 0.0 or image_cancer_risk > 0.0 else ""

        # Using shorten with html.escape for safety
        summary_display = shorten(summary, width=1000, placeholder="...") if summary else "No summary available."

        return """
        <div class="{}">
           
            <div class="{}">
                <div class="{}">
                    <h3 class="mb-0"><i class="bi bi-clipboard-pulse"></i> AI Clinical Analysis - Note ID {}</h3>
                </div>
                <div class="card-body">
                    <div class="mb-4">
                        <h5 class="{}">Search Report</h5>
                        <label for="reportSearch" class="form-label visually-hidden">Search Report</label>
                        <input type="text" class="form-control" id="reportSearch" placeholder="Search keywords, entities, or diagnoses..." onkeyup="filterReport()" aria-label="Search report content">
                    </div>
                    
                    {}
                    {}
                    {}
                    {}
                    {}
                    <div class="mb-4">
                        <h5 class="{}"><i class="bi bi-tags"></i> Keywords</h5>
                        <div class="d-flex flex-wrap gap-2">
                            {}
                        </div>
                    </div>
                    <div class="mb-4">
                        <h5 class="{}"><i class="bi bi-code"></i> CUIs</h5>
                        <div class="d-flex flex-wrap gap-2">
                            {}
                        </div>
                    </div>
                    {}
                    {}
                    <div class="{}">
                        <div class="card-body">
                            <h5 class="card-title {}"><i class="bi bi-file-text"></i> Summary</h5>
                            <p class="card-text text-wrap">{}</p>
                        </div>
                    </div>
                    <div class="d-flex gap-2">
                        <button class="btn btn-primary" onclick="window.print()" aria-label="Download report as PDF"><i class="bi bi-printer"></i> Download as PDF</button>
                        <button class="btn btn-outline-secondary" onclick="copyToClipboard()" aria-label="Copy report to clipboard"><i class="bi bi-clipboard"></i> Copy Report</button>
                    </div>
                </div>
            </div>
        </div>
                <script>
            let timeout;
            function filterReport() {{
                clearTimeout(timeout);
                timeout = setTimeout(() => {{
                    const input = document.getElementById('reportSearch').value.toLowerCase();
                    const elements = document.querySelectorAll('.card-body p, .card-body li, .alert p, .badge');
                    elements.forEach(el => {{
                        const text = el.innerText.toLowerCase();
                        el.style.display = text.includes(input) ? '' : 'none';
                    }});
                }}, 300);
            }}
            function copyToClipboard() {{
                const content = document.querySelector('.card-header.bg-primary + .card-body').innerText;
                if (navigator.clipboard) {{
                    navigator.clipboard.writeText(content).then(() => {{
                        const toast = document.createElement('div');
                        toast.className = 'toast align-items-center text-bg-success border-0 position-fixed bottom-0 end-0 m-3';
                        toast.innerHTML = `
                            <div class="d-flex">
                                <div class="toast-body">Report copied to clipboard!</div>
                                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                            </div>
                        `;
                        document.body.appendChild(toast);
                        const bsToast = new bootstrap.Toast(toast);
                        bsToast.show();
                        setTimeout(() => toast.remove(), 3000);
                    }}).catch(() => {{
                        alert('Failed to copy report.');
                    }});
                }} else {{
                    const textarea = document.createElement('textarea');
                    textarea.value = content;
                    document.body.appendChild(textarea);
                    textarea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textarea);
                    const toast = document.createElement('div');
                    toast.className = 'toast align-items-center text-bg-success border-0 position-fixed bottom-0 end-0 m-3';
                    toast.innerHTML = `
                        <div class="d-flex">
                            <div class="toast-body">Report copied to clipboard!</div>
                            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
                        </div>
                    `;
                    document.body.appendChild(toast);
                    const bsToast = new bootstrap.Toast(toast);
                    bsToast.show();
                    setTimeout(() => toast.remove(), 3000);
                }}
            }}
            document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {{
                new bootstrap.Tooltip(el);
            }});
        </script>

        """.format(
            BOOTSTRAP_CLASSES['container'],
            BOOTSTRAP_CLASSES['card'],
            BOOTSTRAP_CLASSES['card_header'],
            note_id,
            BOOTSTRAP_CLASSES['section_heading'],
            metadata_html,
            primary_html,
            differential_html,
            lab_abnormalities_html,
            risk_scores_html,
            BOOTSTRAP_CLASSES['section_heading'],
            "".join('<span class="{}">{}</span>'.format(BOOTSTRAP_CLASSES['badge_keyword'], keyword) for keyword in keywords),
            BOOTSTRAP_CLASSES['section_heading'],
            "".join('<span class="{}" data-bs-toggle="tooltip" title="Unified Medical Language System CUI">{}</span>'.format(BOOTSTRAP_CLASSES['badge_cui'], cui) for cui in cuis),
            entities_html,
            management_html,
            BOOTSTRAP_CLASSES['card'],
            BOOTSTRAP_CLASSES['section_heading'],
            summary_display
        )
    except Exception as e:
        logger.error(f"HTML generation error: {e}", exc_info=True)
        return """
        <div class="{}">
            <div class="alert alert-danger" role="alert">
                <h4 class="alert-heading">Error Generating Report</h4>
                <p>An unexpected error occurred. Please try again later.</p>
                <p>Contact <a href="mailto:support@example.com">support@example.com</a> for assistance.</p>
            </div>
        </div>""".format(BOOTSTRAP_CLASSES['container'])

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
    """Generate a concise and natural clinical summary, incorporating key SOAP note fields and lab abnormalities."""
    include_fields = include_fields or ['symptoms', 'findings', 'recommendations', 'lab_abnormalities']
    if not text:
        return ""
    
    start_time = time.time()
    if not doc:
        doc = DiseasePredictor.nlp(text)
    
    invalid_terms = {'mg', 'ms', 'g', 'ml', 'mm', 'ng', 'dl', 'hr'}  # Filter out units and invalid terms
    symptoms = set()
    findings = []
    recommendations = []
    lab_abnormalities = soap_note.get('lab_abnormalities', [])
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
                if term in sent_text.lower() and term not in invalid_terms:
                    symptoms.add(term)
    
    if 'findings' in include_fields and soap_note.get('assessment'):
        for term in DiseasePredictor.clinical_terms:
            if term in soap_note['assessment'].lower() and term not in invalid_terms:
                findings.append(term)
    
    if 'recommendations' in include_fields and soap_note.get('recommendation'):
        recommendations.append(soap_note['recommendation'])
    
    if 'lab_abnormalities' in include_fields and lab_abnormalities:
        lab_abnormalities = [lab for lab in lab_abnormalities if lab.lower() not in invalid_terms]
    
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
    
    if lab_abnormalities and 'lab_abnormalities' in include_fields:
        labs_str = humanize_list(sorted(lab_abnormalities))
        summary_parts.append(f"Lab abnormalities include {labs_str}.")
    
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
    
    return summary if summary else text[:200] + "..."