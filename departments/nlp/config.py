import torch

# Model and Device Configuration
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Processing Parameters
MAX_LENGTH = 512
BATCH_SIZE = 8
EMBEDDING_DIM = 768
SIMILARITY_THRESHOLD = 0.9
CONFIDENCE_THRESHOLD = 0.85

# departments/nlp/config.py
UTS_API_KEY = "c7c9be68-bfa2-4fe6-850c-11ed2136a253"
UTS_BASE_URL = "https://uts-ws.nlm.nih.gov/rest"


