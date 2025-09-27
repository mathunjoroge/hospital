import sys
import os
sys.path.append('/home/mathu/projects/hospital')

from nlp import DiseasePredictor

predictor = DiseasePredictor()
predictor.initialize()
text = "Patient with resistant bacterial infection and inadequate infection control measures."
result = predictor.predict_amr_ipc(text)
print("AMR/IPC Prediction:", result)