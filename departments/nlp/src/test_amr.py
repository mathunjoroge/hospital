import sys
import os
sys.path.append('/home/mathu/projects/hospital')

from departments.nlp.src.nlp import DiseasePredictor

def test_amr_prediction():
    predictor = DiseasePredictor()
    predictor.initialize()
    text = "Patient with resistant bacterial infection and inadequate infection control measures."
    result = predictor.predict_amr_ipc(text)
    assert isinstance(result, dict)
    print("AMR/IPC Prediction:", result)

if __name__ == "__main__":
    test_amr_prediction()