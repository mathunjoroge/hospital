from functools import lru_cache
import logging
from typing import List, Dict, Tuple, FrozenSet
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from sqlalchemy.sql import text
from sqlalchemy.orm import Session

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

# Use a frozenset for immutable, slightly faster lookups.
SYMPTOM_RELATIONSHIPS: FrozenSet[str] = frozenset([
    'manifestation_of', 'has_finding', 'has_sign_or_symptom',
    'indication_of', 'symptom_of', 'associated_with',
    'finding_site_of', 'due_to',
])

class DiseaseSymptomMapper:
    """
    Maps diseases to symptoms and vice versa using UMLS relationships.
    
    This optimized version uses parameterized SQL queries for performance and
    security, and refactors duplicated logic into a shared helper method.
    """
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls) -> 'DiseaseSymptomMapper':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        # Removed unused self.cache from cachetools
        self.umls_mapper = UMLSMapper.get_instance()
        self._initialized = True

    def _query_umls_relations(self, cui: str, direction: str) -> List[Dict]:
        """
        Private helper to query UMLS for related concepts, avoiding code duplication.
        
        Args:
            cui: The Concept Unique Identifier to query for.
            direction: 'symptoms' (finds symptoms for a disease) or 
                       'diseases' (finds diseases for a symptom).
        """
        if direction == 'symptoms':
            cui1_col, cui2_col = 'r.cui1', 'r.cui2'
            name_col, result_cui_col = 'c2.str', 'c2.cui'
            result_keys = ('symptom_name', 'symptom_cui')
        elif direction == 'diseases':
            cui1_col, cui2_col = 'r.cui1', 'r.cui2'
            name_col, result_cui_col = 'c1.str', 'c1.cui'
            result_keys = ('disease_name', 'disease_cui')
        else:
            raise ValueError("Invalid direction specified.")

        # CRUCIAL: Use bind parameters (:param) instead of f-strings.
        # This is faster (query plan caching) and prevents SQL injection.
        # SQLAlchemy handles the tuple expansion for the 'IN' clause automatically.
        query = text(f"""
            SELECT DISTINCT {name_col} AS name, {result_cui_col} AS cui
            FROM umls.mrrel r
            JOIN umls.mrconso c1 ON r.cui1 = c1.cui
            JOIN umls.mrconso c2 ON r.cui2 = c2.cui
            WHERE {cui1_col if direction == 'symptoms' else cui2_col} = :cui
                AND r.rela IN :relationships
                AND c1.lat = :language AND c1.suppress = 'N'
                AND c2.lat = :language AND c2.suppress = 'N'
                AND c1.sab IN :trusted_sources
                AND c2.sab IN :trusted_sources
        """)
        
        try:
            with UMLSSession() as session:
                results = session.execute(
                    query,
                    {
                        'cui': cui,
                        'relationships': tuple(SYMPTOM_RELATIONSHIPS),
                        'language': HIMS_CONFIG["UMLS_LANGUAGE"],
                        'trusted_sources': tuple(HIMS_CONFIG["TRUSTED_SOURCES"])
                    }
                ).fetchall()
                
                logger.debug(f"Found {len(results)} {direction} for CUI {cui}")
                return [{'name': row.name, 'cui': row.cui} for row in results]
        except Exception as e:
            logger.error(f"Error fetching {direction} for CUI {cui}: {e}")
            return []

    @lru_cache(maxsize=1000)
    def get_disease_symptoms(self, disease_cui: str) -> List[Dict]:
        """Gets symptoms for a given disease CUI."""
        return self._query_umls_relations(disease_cui, 'symptoms')
    
    @lru_cache(maxsize=1000)
    def get_symptom_diseases(self, symptom_cui: str) -> List[Dict]:
        """Gets diseases for a given symptom CUI."""
        return self._query_umls_relations(symptom_cui, 'diseases')
    
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
                # Pass the mapper instance to avoid repeated `get_instance` calls
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
    
    def _process_disease_signature(self, disease_id: int, disease_name: str, umls_mapper: UMLSMapper) -> Tuple[str, set]:
        symptoms = set()
        try:
            disease_cui = None
            # The pattern of creating a new connection per thread is generally safe for SQLite
            # and necessary for thread-safety in many database drivers.
            with get_sqlite_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cui FROM disease_keywords WHERE disease_id = ? LIMIT 1", (disease_id,))
                row = cursor.fetchone()
                if row:
                    disease_cui = row['cui']
                
                if not disease_cui:
                    disease_cuis = umls_mapper.map_term_to_cui(disease_name)
                    if disease_cuis:
                        disease_cui = disease_cuis[0]
                
                if disease_cui:
                    symptom_data = self.get_disease_symptoms(disease_cui)
                    symptoms.update(symptom['name'].lower() for symptom in symptom_data)
                
                # Fallback to local DB mapping if UMLS yields no results
                if not symptoms:
                    cursor.execute("""
                        SELECT s.name 
                        FROM disease_symptoms ds
                        JOIN symptoms s ON ds.symptom_id = s.id
                        WHERE ds.disease_id = ?
                    """, (disease_id,))
                    symptoms.update(row['name'].lower() for row in cursor.fetchall())
        except Exception as e:
            logger.error(f"Error processing disease signature for {disease_name}: {e}")
        
        return disease_name, symptoms