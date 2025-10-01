# test_summarizer.py
from summarizer import ClinicalSummarizer

def run_test():
    print("=== Testing ClinicalSummarizer ===")
    
    summarizer = ClinicalSummarizer()
    
    # Test with a structured dictionary
    sample_note = {
        "hpi": "Patient is a 60-year-old woman presenting with chest pain and shortness of breath for 2 days.",
        "medical_history": "History of hypertension and obesity. No known allergies.",
        "assessment": "Likely unstable angina. Needs further evaluation.",
        "recommendation": "Admit to hospital. Start aspirin, oxygen, and monitor vitals. Order ECG and troponin levels."
    }
    
    print("\n--- Test with structured dict ---")
    summary1 = summarizer.summarize(sample_note)
    print(summary1)

    # Test with plain text
    sample_text = (
        "A 45-year-old male presents with fever, cough, and fatigue after exposure to a sick contact. "
        "Past history includes diabetes and hypertension. Assessment: possible viral infection. "
        "Plan: order influenza test and provide supportive care."
    )
    
    print("\n--- Test with plain text ---")
    summary2 = summarizer.summarize(sample_text)
    print(summary2)

    # Test with invalid input
    print("\n--- Test with invalid input ---")
    summary3 = summarizer.summarize(12345)  # not a dict or string
    print(summary3)

if __name__ == "__main__":
    run_test()
