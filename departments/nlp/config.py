import torch

# Model and Device Configuration
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Processing Parameters
MAX_LENGTH = 512
BATCH_SIZE = 16
EMBEDDING_DIM = 768
SIMILARITY_THRESHOLD = 0.9
CONFIDENCE_THRESHOLD = 0.85