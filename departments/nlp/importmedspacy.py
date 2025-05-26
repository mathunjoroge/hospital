from departments.models.medicine import SOAPNote
from departments.nlp.ai_summary import generate_ai_analysis
note = SOAPNote(
    id=1,
    situation="patient complains of headache",
    hpi="the headache started three days ago. the headache is associated with fevers and chills, nausea and vomiting and loss of appetite",
    aggravating_factors="taking foods with fats makes the nausea worse",
    alleviating_factors="taking strong tea and licking salt alleviates the nausea",
    medical_history="no chronic illness reported",
    medication_history="patient takes paracetamol 1000 mg for headache",
    assessment="the patient has jaundice in the eyes",
    recommendation="peripheral blood smear for malaria parasite"
)
print(generate_ai_analysis(note))