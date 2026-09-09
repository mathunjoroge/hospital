"""
tests/test_nvidia_nim.py
────────────────────────
Unit tests for NvidiaNIMClient (departments/nlp/src/nvidia_client.py).
Exercises offline rule-based fallbacks for cancer risk prediction,
AMR/IPC risk scoring, clinical note summarization, and DICOM radiology analysis.
"""

from departments.nlp.src.nvidia_client import NvidiaNIMClient


def test_client_initialization():
    """Test NvidiaNIMClient initializes cleanly with default model."""
    client = NvidiaNIMClient()
    assert client is not None
    assert client.model == "meta/llama-3.2-11b-vision-instruct"


def test_cancer_risk_prediction_fallback():
    """Test offline rule-based fallback for cancer risk prediction."""
    client = NvidiaNIMClient()
    note_text = "Patient presents with a persistent lump in breast, nipple discharge, and abnormal mammogram."
    res = client.predict_cancer_risk(note_text)

    assert isinstance(res, dict)
    assert "breast cancer" in res
    max_cancer = max(res, key=res.get)
    assert max_cancer == "breast cancer"


def test_amr_ipc_prediction_high_risk():
    """Test AMR/IPC prediction for high resistance keywords."""
    client = NvidiaNIMClient()
    note_text = "Patient diagnosed with MRSA bacteremia. Staff noted unwashed hands and PPE breach."
    res = client.predict_amr_ipc(note_text)

    assert isinstance(res, dict)
    assert res["amr_high"] > 0.5
    assert res["ipc_inadequate"] > 0.5


def test_amr_ipc_prediction_adequate_ipc():
    """Test AMR/IPC prediction with proper isolation protocol."""
    client = NvidiaNIMClient()
    note_text = "Patient placed in strict isolation with PPE, glove, mask, and hand hygiene observed."
    res = client.predict_amr_ipc(note_text)

    assert isinstance(res, dict)
    assert res["ipc_adequate"] > 0.5


def test_summarize_note():
    """Test clinical note summarization fallback."""
    client = NvidiaNIMClient()
    note_text = "A 55-year-old male presents with severe chest pain radiating to the left arm. History of hypertension. Assessment: Acute Coronary Syndrome. Plan: ECG, troponin, aspirin."
    summary = client.summarize_note(note_text)

    assert isinstance(summary, str)
    assert len(summary) > 0


def test_analyze_radiology():
    """Test DICOM radiology AI analysis fallback."""
    client = NvidiaNIMClient()
    res = client.analyze_radiology(
        modality="CT",
        body_part="Chest",
        description="Rule out pulmonary embolism",
        symptoms="Acute dyspnea and pleuritic chest pain",
    )

    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "predictions" in res
    assert "confidence" in res
    assert "impression" in res
