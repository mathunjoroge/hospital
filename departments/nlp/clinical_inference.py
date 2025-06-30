import spacy
from departments.nlp.nlp_pipeline import get_postgres_connection
from nltk.stem import WordNetLemmatizer
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)
lemmatizer = WordNetLemmatizer()

class SciBERTWrapper:
    def __init__(self, model_name="en_core_sci_sm", disable_linker=True):
        try:
            self.nlp = spacy.load(model_name, disable=["lemmatizer"])
            if disable_linker and "entity_linker" in self.nlp.pipe_names:
                self.nlp.remove_pipe("entity_linker")
            if self.nlp.has_pipe("ner"):
                logger.info(f"NER component available in model: {model_name}")
            else:
                logger.warning("NER not available. Entity extraction will be ineffective.")
        except OSError as e:
            logger.error(f"Failed to load SpaCy model {model_name}: {e}. Using blank model.")
            self.nlp = spacy.blank("en")

    def extract_entities(self, text):
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

_sci_ner = SciBERTWrapper()

def predict_diseases(note: str) -> List[Dict]:
    # Step 1: Extract and normalize symptoms
    entities = _sci_ner.extract_entities(note)
    terms = sorted(set(
        lemmatizer.lemmatize(ent.lower().strip()) for ent, _ in entities
    ))
    if not terms:
        logger.warning("No entities found in the note.")
        return []

    # Step 2: Get CUIs for symptoms
    symptom_cuis = []
    with get_postgres_connection(readonly=True) as cursor:
        for term in terms:
            cursor.execute("""
                SELECT DISTINCT c.cui
                FROM umls.mrconso c
                JOIN umls.mrsty s ON c.cui = s.cui
                WHERE c.str ILIKE %s
                  AND c.lat = 'ENG'
                  AND s.sty IN ('Sign or Symptom', 'Finding', 'Pathologic Function')
                LIMIT 1
            """, (f"%{term}%",))
            res = cursor.fetchone()
            if res:
                logger.info(f"🧠 Term '{term}' → CUI {res['cui']}")
                symptom_cuis.append(res['cui'])
            else:
                logger.info(f"⚠️ Term '{term}' → No CUI found")

    if len(symptom_cuis) < 3:
        logger.warning("Too few CUIs found for reliable disease prediction.")
        return []

    # Step 3: Use CUIs to find related diseases
    with get_postgres_connection(readonly=True) as cursor:
        cursor.execute("""
            WITH symptom_cuis AS (
                SELECT DISTINCT unnest(%s::text[]) AS cui
            ),
            disease_candidates AS (
                SELECT r.cui2 AS disease_cui, d.str AS disease
                FROM umls.mrrel r
                JOIN symptom_cuis s ON r.cui1 = s.cui
                JOIN umls.mrconso d ON r.cui2 = d.cui
                JOIN umls.mrsty st ON r.cui2 = st.cui
                WHERE r.rel IN ('RO', 'RB', 'SY', 'RIN', 'RN', 'CHD', 'PAR')
                  AND d.lat = 'ENG'
                  AND d.ts = 'P'
                  AND d.ispref = 'Y'
                  AND st.sty = 'Disease or Syndrome'
            ),
            disease_count AS (
                SELECT disease_cui, disease, COUNT(*) AS matched_symptoms
                FROM disease_candidates
                GROUP BY disease_cui, disease
            )
            SELECT disease_cui, disease, matched_symptoms
            FROM disease_count
            ORDER BY matched_symptoms DESC
            LIMIT 50;
        """, (symptom_cuis,))
        diseases = cursor.fetchall()

    # Step 4: Boost based on raw text mention
    boosted = []
    for row in diseases:
        name = row["disease"]
        score = row["matched_symptoms"]
        if name.lower() in note.lower():
            score += 5
            logger.info(f"✨ Boosted '{name}' due to mention in note.")
        boosted.append({
            "name": name,
            "cui": row["disease_cui"],
            "score": score,
        })

    return sorted(boosted, key=lambda x: -x["score"])[:10]
