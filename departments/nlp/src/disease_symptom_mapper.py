from functools import lru_cache
import logging
from typing import List, Dict
from collections import defaultdict
from cachetools import LRUCache
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from sqlalchemy.sql import text

# Local project imports
from src.database import get_sqlite_connection, UMLSSession
from src.config import get_config
from departments.nlp.resources.cancer_diseases import cancer_diseases
from resources.common_fallbacks import COMMON_SYMPTOM_DISEASE_MAP
from .umls_mapper import UMLSMapper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HIMS-NLP")
HIMS_CONFIG = get_config()

# Enhanced UMLS relationship types
SYMPTOM_RELATIONSHIPS = [
    'manifestation_of',
    'has_finding',
    'has_sign_or_symptom',
    'indication_of',
    'symptom_of',
    'associated_with',
    'finding_site_of',
    'due_to',
]

class DiseaseSymptomMapper:
    """Maps diseases to symptoms and vice versa using UMLS relationships."""
    
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls) -> 'DiseaseSymptomMapper':
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.cache = LRUCache(maxsize=1000)
        self.umls_mapper = UMLSMapper.get_instance()
        self._initialized = True
    
    @lru_cache(maxsize=1000)
    def get_disease_symptoms(self, disease_cui: str) -> List[Dict]:
        try:
            with UMLSSession() as session:
                query = text(f"""
                    SELECT DISTINCT c2.str AS symptom_name, c2.cui AS symptom_cui
                    FROM umls.mrrel r
                    JOIN umls.mrconso c1 ON r.cui1 = c1.cui
                    JOIN umls.mrconso c2 ON r.cui2 = c2.cui
                    WHERE r.cui1 = :disease_cui
                        AND r.rela IN ({', '.join([f"'{rel}'" for rel in SYMPTOM_RELATIONSHIPS])})
                        AND c1.lat = :language AND c1.suppress = 'N'
                        AND c2.lat = :language AND c2.suppress = 'N'
                        AND c1.sab IN :trusted_sources
                        AND c2.sab IN :trusted_sources
                """)
                symptoms = session.execute(
                    query,
                    {
                        'disease_cui': disease_cui,
                        'language': HIMS_CONFIG["UMLS_LANGUAGE"],
                        'trusted_sources': tuple(HIMS_CONFIG["TRUSTED_SOURCES"])
                    }
                ).fetchall()
                
                logger.debug(f"Found {len(symptoms)} symptoms for disease CUI {disease_cui}")
                return [{'name': row.symptom_name, 'cui': row.symptom_cui} for row in symptoms]
        except Exception as e:
            logger.error(f"Error fetching symptoms for disease CUI {disease_cui}: {e}")
            return []
    
    @lru_cache(maxsize=1000)
    def get_symptom_diseases(self, symptom_cui: str) -> List[Dict]:
        try:
            with UMLSSession() as session:
                query = text(f"""
                    SELECT DISTINCT c1.str AS disease_name, c1.cui AS disease_cui
                    FROM umls.mrrel r
                    JOIN umls.mrconso c1 ON r.cui1 = c1.cui
                    JOIN umls.mrconso c2 ON r.cui2 = c2.cui
                    WHERE r.cui2 = :symptom_cui
                        AND r.rela IN ({', '.join([f"'{rel}'" for rel in SYMPTOM_RELATIONSHIPS])})
                        AND c1.lat = :language AND c1.suppress = 'N'
                        AND c2.lat = :language AND c2.suppress = 'N'
                        AND c1.sab IN :trusted_sources
                        AND c2.sab IN :trusted_sources
                """)
                diseases = session.execute(
                    query,
                    {
                        'symptom_cui': symptom_cui,
                        'language': HIMS_CONFIG["UMLS_LANGUAGE"],
                        'trusted_sources': tuple(HIMS_CONFIG["TRUSTED_SOURCES"])
                    }
                ).fetchall()
                
                logger.debug(f"Found {len(diseases)} diseases for symptom CUI {symptom_cui}")
                return [{'name': row.disease_name, 'cui': row.disease_cui} for row in diseases]
        except Exception as e:
            logger.error(f"Error fetching diseases for symptom CUI {symptom_cui}: {e}")
            return []
    
    def get_symptom_diseases_fallback(self, symptom_text: str) -> List[str]:
        normalized = self.umls_mapper.normalize_symptom(symptom_text)
        return COMMON_SYMPTOM_DISEASE_MAP.get(normalized, [])
    
    def build_disease_signatures(self) -> Dict[str, set]:
        disease_signatures = defaultdict(set)
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, name FROM diseases")
                diseases = cursor.fetchall()
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self._process_disease_signature, disease_id, disease_name, self.umls_mapper)
                    for disease_id, disease_name in diseases
                ]
                
                for future in futures:
                    disease_name, symptoms = future.result()
                    if symptoms:
                        disease_signatures[disease_name] = symptoms
            
            for disease, data in cancer_diseases.items():
                disease_signatures[disease].update(data['symptoms'])
            
            logger.info(f"Built disease signatures for {len(disease_signatures)} diseases")
            return disease_signatures
        except Exception as e:
            logger.error(f"Error building disease signatures: {e}")
            return {}
    
    def _process_disease_signature(self, disease_id, disease_name, umls_mapper):
        symptoms = set()
        try:
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cui FROM disease_keywords WHERE disease_id = ? LIMIT 1", (disease_id,))
                row = cursor.fetchone()
                disease_cui = row['cui'] if row else None
                
                if not disease_cui:
                    disease_cuis = umls_mapper.map_term_to_cui(disease_name)
                    disease_cui = disease_cuis[0] if disease_cuis else None
                
                if disease_cui:
                    symptom_data = self.get_disease_symptoms(disease_cui)
                    for symptom in symptom_data:
                        symptoms.add(symptom['name'].lower())
                
                if not symptoms:
                    cursor.execute("""
                        SELECT s.name 
                        FROM disease_symptoms ds
                        JOIN symptoms s ON ds.symptom_id = s.id
                        WHERE ds.disease_id = ?
                    """, (disease_id,))
                    for row in cursor.fetchall():
                        symptoms.add(row['name'].lower())
        except Exception as e:
            logger.error(f"Error processing disease signature for {disease_name}: {e}")
        
        return disease_name, symptoms