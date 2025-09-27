import logging
import os
from datetime import datetime
from departments.nlp.src.nlp import DiseasePredictor
from departments.nlp.src.utils import generate_html_response

# Configure logging
logging.basicConfig(
    filename='test_note.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HIMS-NLP-TEST")

def test_process_soap_note():
    """Test the processing of a sample SOAP note."""
    logger.info("Starting test_process_soap_note...")
    
    # Initialize DiseasePredictor
    try:
        DiseasePredictor.initialize()
        logger.info("DiseasePredictor initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize DiseasePredictor: {e}", exc_info=True)
        raise
    
    # Create a sample SOAP note
    sample_note = {
        "id": 1001,  # Integer primary key
        "patient_id": "P1000",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "situation": "Patient presents with persistent fever, cough, and fatigue following recent hospitalization for a urinary tract infection.",
        "hpi": "45-year-old male with a 5-day history of fever (up to 39°C), productive cough with yellow sputum, and increasing fatigue. Reports shortness of breath on exertion. Recently treated for a urinary tract infection with ciprofloxacin 500 mg BID for 7 days, completed 2 weeks ago. Symptoms began 3 days after completing antibiotics.",
        "symptoms": "fever, productive cough, yellow sputum, fatigue, shortness of breath",
        "aggravating_factors": "Symptoms worsen with physical activity and at night.",
        "alleviating_factors": "Rest and hydration provide mild relief.",
        "medical_history": "Type 2 diabetes mellitus (diagnosed 2018, controlled with metformin), history of recurrent urinary tract infections, hypertension, smoking (10 pack-years, quit 2 years ago).",
        "medication_history": "Ciprofloxacin 500 mg BID for 7 days (completed 2 weeks ago), metformin 1000 mg BID, lisinopril 10 mg daily.",
        "assessment": "Suspected post-infectious pneumonia, possible antibiotic-resistant organism given recent ciprofloxacin use. Consider community-acquired pneumonia vs. healthcare-associated pneumonia. Rule out early sepsis and malignancy given fatigue and smoking history.",
        "recommendation": "Order chest X-ray, sputum culture, and blood cultures. Start empiric antibiotics (ceftriaxone 1g IV daily + azithromycin 500 mg PO daily). Monitor for sepsis. Consider CT chest if no improvement in 48 hours. Refer to infectious disease if cultures indicate resistance.",
        "additional_notes": "Patient advised to maintain hydration and monitor symptoms. Discussed smoking cessation benefits given history.",
        "ai_notes": "",  # Initially empty, to be populated by NLP pipeline
        "ai_analysis": "",  # Initially empty, to be populated by NLP pipeline
        "file_path": "/home/mathu/projects/hospital/uploads/P1000_note_1001.pdf"  # Example path
    }
    
    # Process the note
    predictor = DiseasePredictor()
    logger.info(f"Processing SOAP note ID {sample_note['id']}...")
    start_time = datetime.now()
    result = predictor.process_soap_note(sample_note)
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # Validate result
    assert "error" not in result, f"Processing failed: {result.get('details', 'Unknown error')}"
    assert result.get("note_id") == sample_note["id"], "Note ID mismatch"
    assert result.get("patient_id") == sample_note["patient_id"], "Patient ID mismatch"
    assert "primary_diagnosis" in result, "Primary diagnosis missing"
    assert "differential_diagnoses" in result, "Differential diagnoses missing"
    assert "amr_ipc_probabilities" in result, "AMR/IPC probabilities missing"
    assert "amr_ipc_recommendations" in result, "AMR/IPC recommendations missing"
    
    logger.info(f"Processing completed in {processing_time:.3f} seconds.")
    logger.info(f"AMR/IPC Probabilities: {result['amr_ipc_probabilities']}")
    logger.info(f"AMR/IPC Recommendations: {result['amr_ipc_recommendations']}")
    
    # Generate HTML report
    logger.info("Generating HTML report...")
    html_output = generate_html_response(result)
    
    # Save HTML output
    output_file = "test_note_output.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_output)
    logger.info(f"HTML report saved to {output_file}")
    
    # Verify AMR/IPC content in HTML
    assert any(term in html_output.lower() for term in ["amr/ipc analysis", "amr_high", "ipc_adequate"]), "AMR/IPC content missing in HTML"
    
    logger.info("All tests passed successfully!")

if __name__ == "__main__":
    try:
        test_process_soap_note()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise