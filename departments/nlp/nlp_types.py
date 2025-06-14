# departments/nlp/nlp_types.py
from enum import Enum

class Category(Enum):
    """
    Enum for categorizing medical terms, symptoms, and diagnoses in the NLP pipeline.
    Each category represents a medical domain used across the system for consistent
    in symptom tracking, clinical analysis, and knowledge base initialization.
    """
    GENERAL = "general"
    """General symptoms or conditions not specific to a medical domain."""
    CARDIOVASCULAR = "cardiovascular"
    """Heart and blood vessel-related conditions."""
    NEUROLOGICAL = "neurological"
    """Nervous system-related conditions."""
    RESPIRATORY = "respiratory"
    """Lung and breathing-related conditions."""
    GASTROINTESTINAL = "gastrointestinal"
    """Digestive system-related conditions."""
    INFECTIOUS = "infectious"
    """Conditions caused by pathogens."""
    DERMATOLOGICAL = "dermatological"
    """Skin-related conditions."""
    MUSCULOSKELETAL = "musculoskeletal"
    """Bone, muscle, and joint-related conditions."""
    HEPATIC = "hepatic"
    """Liver-related conditions."""
    ENDOCRINOLOGY = "endocrinology"
    """Endocrine system-related conditions."""
    ONCOLOGY = "oncology"
    """Cancer-related conditions."""
    PSYCHIATRY = "psychiatry"
    """
    Psychiatric and mental health-related conditions.
    """