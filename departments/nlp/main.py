import sys
import nltk
from src.nlp import DiseasePredictor
from src.cli import HIMSCLI
from src.api import start_server

if __name__ == '__main__':
    nltk.download('wordnet', quiet=True)
    DiseasePredictor.initialize()
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        HIMSCLI().run()
    else:
        start_server()