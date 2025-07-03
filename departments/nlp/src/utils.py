import re
import html
from typing import Dict, List, Tuple
from collections import defaultdict
import time
from datetime import datetime
import pytz
import spacy
from src.nlp import DiseasePredictor
from resources.priority_symptoms import PRIORITY_SYMPTOMS
import logging

logger = logging.getLogger("HIMS-NLP")

def generate_primary_diagnosis_html(primary_diagnosis: dict) -> str:
    """Generate HTML for primary diagnosis using Bootstrap classes."""
    if not primary_diagnosis:
        return """
        <div class="alert alert-danger" role="alert">
            <h5 class="alert-heading">Primary Diagnosis</h5>
            <p>No primary diagnosis identified</p>
        </div>
        """
    
    return f"""
    <div class="alert alert-success" role="alert">
        <h5 class="alert-heading">Primary Diagnosis</h5>
        <p><strong>Disease:</strong> {html.escape(primary_diagnosis.get('disease', 'N/A'))}</p>
        <p><span class="badge bg-success">Score: {primary_diagnosis.get('score', 'N/A')}</span></p>
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
    <div class="mb-3">
        <h5>Differential Diagnoses</h5>
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
    
    entity_items = ""
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
                {'<li><button class="btn btn-link btn-sm show-more-btn" data-entity="{html.escape(text)}">Show More ({more_count})</button></li>' if more_count > 0 else ''}
            </ul>
        </div>
        """
        
        entity_items += f"""
        <div class="card mb-2 {'border-warning bg-warning-subtle' if is_priority else ''}">
            <div class="card-header d-flex justify-content-between align-items-center">
                <div>
                    <strong>{html.escape(text)}</strong>
                    <span class="badge bg-primary ms-2">{html.escape(label)}</span>
                    {'<span class="badge bg-danger ms-2">Priority</span>' if is_priority else ''}
                </div>
                <button class="btn btn-link btn-sm" type="button" data-bs-toggle="collapse" data-bs-target="#entity-{html.escape(text.replace(' ', '-'))}">
                    Toggle Details
                </button>
            </div>
            <div id="entity-{html.escape(text.replace(' ', '-'))}" class="collapse">
                <div class="card-body">
                    <p class="small">Severity: {context["severity"]}, Temporal: {html.escape(context["temporal"])}</p>
                    {disease_list if diseases else '<p>No associated diseases</p>'}
                </div>
            </div>
        </div>
        """
    
    return f"""
    <div class="mb-3">
        <h5>Entities</h5>
        <div>
            {entity_items}
        </div>
    </div>
    """

def generate_management_plans_html(management_plans: dict) -> str:
    """Generate HTML for management plans using Bootstrap classes."""
    if not management_plans:
        return ""
    
    plans_html = "".join(
        f'<div class="card mb-2"><div class="card-body"><strong>{html.escape(disease)}:</strong> {html.escape(plan)}</div></div>'
        for disease, plan in management_plans.items()
    )
    return f"""
    <div class="mb-3">
        <h5>Management Plans</h5>
        <div>
            {plans_html}
        </div>
    </div>
    """

def generate_html_response(data: Dict[str, any], status_code: int = 200) -> str:
    """Generate HTML div content for the /process_note endpoint using Bootstrap classes."""
    if status_code == 404:
        return """
        <div class="container">
            <div class="alert alert-danger" role="alert">
                <h5 class="alert-heading">Error: Note Not Found</h5>
                <p>The requested note was not found in the database.</p>
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
    # Use created_at from SOAP note if available, otherwise current time
    created_at = data.get("created_at")
    if created_at:
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        processed_at = created_at.astimezone(pytz.timezone('Africa/Nairobi')).strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        processed_at = datetime.now(pytz.timezone('Africa/Nairobi')).strftime("%Y-%m-%d %H:%M:%S %Z")
    processed_at = html.escape(processed_at)
    processing_time = data.get("processing_time", 0.0)

    primary_html = generate_primary_diagnosis_html(primary_diagnosis)
    differential_html = generate_differential_diagnoses_html(differential_diagnoses)
    entities_html = generate_entities_html(entities)
    management_html = generate_management_plans_html(management_plans)

    # Deduplicate and limit summary sentences
    summary_sentences = summary.split(". ")
    unique_sentences = []
    seen = set()
    for sentence in summary_sentences:
        if sentence and sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)
    summary = ". ".join(unique_sentences[:4]) + ('.' if unique_sentences else '')

    return f"""
    <div class="container">
        <h3 class="mb-3 text-primary">🧠 AI Clinical Analysis - Note ID {note_id}</h3>
        
        <div class="row row-cols-1 row-cols-md-3 g-3 mb-3">
            <div class="col">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">Patient ID</h6>
                        <p class="card-text">{patient_id}</p>
                    </div>
                </div>
            </div>
            <div class="col">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">Processed At</h6>
                        <p class="card-text">{processed_at}</p>
                    </div>
                </div>
            </div>
            <div class="col">
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">Processing Time</h6>
                        <p class="card-text">{processing_time:.3f} seconds</p>
                    </div>
                </div>
            </div>
        </div>

        {primary_html}
        {differential_html}

        <div class="mb-3">
            <h5>Keywords</h5>
            <div class="d-flex flex-wrap gap-2">
                {"".join(f'<span class="badge bg-secondary">{keyword}</span>' for keyword in keywords)}
            </div>
        </div>

        <div class="mb-3">
            <h5>CUIs</h5>
            <div class="d-flex flex-wrap gap-2">
                {"".join(f'<span class="badge bg-secondary" data-bs-toggle="tooltip" title="Unified Medical Language System CUI">{cui}</span>' for cui in cuis)}
            </div>
        </div>

        {entities_html}
        {management_html}

        <div class="card mb-3">
            <div class="card-body">
                <h5 class="card-title">Summary</h5>
                <p class="card-text">{summary or 'No summary available.'}</p>
            </div>
        </div>

        <button class="btn btn-primary" onclick="window.print()">Download as PDF</button>
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

def generate_summary(text: str, soap_note: Dict, max_sentences: int = 6, doc=None) -> str:
    """Generate a concise and natural clinical summary, incorporating key SOAP note fields."""
    if not text:
        return ""
    
    start_time = time.time()
    if not doc:
        doc = DiseasePredictor.nlp(text)
    
    clinical_terms = DiseasePredictor.clinical_terms
    temporal_pattern = re.compile(
        r"\b(\d+\s*(day|days|week|weeks|month|months|year|years)\s*(ago)?)\b", 
        re.IGNORECASE
)
    
    # Initialize components
    symptoms = set()
    temporal_info = ""
    aggravating_factors = []
    alleviating_factors = []
    medications = []
    findings = []
    recommendations = []
    medical_history = []
    
    # Directly use structured fields from SOAP note
    if soap_note.get('aggravating_factors'):
        aggravating_factors.append(soap_note['aggravating_factors'])
    if soap_note.get('alleviating_factors'):
        alleviating_factors.append(soap_note['alleviating_factors'])
    if soap_note.get('medication_history'):
        medications.append(soap_note['medication_history'])
    if soap_note.get('medical_history'):
        medical_history.append(soap_note['medical_history'])
    if soap_note.get('recommendation'):
        recommendations.append(soap_note['recommendation'])
    
    # Process each sentence
    text_lower = text.lower()
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_lower = sent_text.lower()
        
        # Temporal information
        if not temporal_info:
            match = temporal_pattern.search(sent_text)
            if match:
                temporal_info = match.group(0)
        
        # Symptoms
        for term in clinical_terms:
            if term in sent_lower and term != 'malaria':
                symptoms.add(term)
        
        # Findings (e.g., jaundice)
        if soap_note.get('assessment') and "jaundice" in sent_lower:
            findings.append("jaundice")
    
    # Build summary with prioritized components
    summary_parts = []
    seen = set()
    
    # 1. Presentation (symptoms + temporal)
    if symptoms:
        symptoms_str = humanize_list(sorted(symptoms))
        presentation = f"The patient presents with {symptoms_str}"
        if temporal_info:
            presentation += f", starting {temporal_info}"
        summary_parts.append(presentation + ".")
        seen.add(presentation)
    
    # 2. Aggravating factors
    for factor in aggravating_factors:
        if factor and "Aggravating factors" not in seen:
            summary_parts.append(f"Aggravating factors: {factor}.")
            seen.add("Aggravating factors")
    
    # 3. Alleviating factors
    for factor in alleviating_factors:
        if factor and "Alleviating factors" not in seen:
            summary_parts.append(f"Alleviating factors: {factor}.")
            seen.add("Alleviating factors")
    
    # 4. Medical history
    for history in medical_history:
        if history and "Medical history" not in seen:
            summary_parts.append(f"Medical history: {history}.")
            seen.add("Medical history")
    
    # 5. Medication history
    for med in medications:
        if med and "Medication history" not in seen:
            summary_parts.append(f"Medication history: {med}.")
            seen.add("Medication history")
    
    # 6. Findings
    for finding in findings:
        if finding and finding not in seen:
            summary_parts.append(f"Notable findings include {finding}.")
            seen.add(finding)
    
    # 7. Recommendations
    for rec in recommendations:
        if rec and rec not in seen:
            summary_parts.append(f"The clinician recommends {rec}.")
            seen.add(rec)
    
    # Limit to max_sentences and clean up
    summary = " ".join(summary_parts[:max_sentences])
    summary = re.sub(r"\s+", " ", summary).strip()
    summary = re.sub(r"\.+", ".", summary).strip()
    
    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    logger.debug(f"Summary generation took {time.time() - start_time:.3f} seconds")
    logger.debug(f"Symptoms: {symptoms}")
    logger.debug(f"Aggravating factors: {aggravating_factors}")
    logger.debug(f"Alleviating factors: {alleviating_factors}")
    logger.debug(f"Medical history: {medical_history}")
    logger.debug(f"Medication history: {medications}")
    logger.debug(f"Findings: {findings}")
    logger.debug(f"Recommendations: {recommendations}")
    logger.debug(f"Generated summary: {summary}")
    
    return summary if summary else text[:200] + "..."