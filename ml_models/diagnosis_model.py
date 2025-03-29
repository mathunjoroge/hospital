# ml_models/diagnosis_model.py
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_diagnosis_model():
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data.data, data.target

    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, 'ml_models/diagnosis_model.pkl')

def predict_diagnosis(data):
    model = joblib.load('ml_models/diagnosis_model.pkl')
    return model.predict([data])[0]