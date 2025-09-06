import logging
import time
from typing import List, Dict
from sqlalchemy.sql import text
from cachetools import LRUCache
from threading import Lock
from collections import defaultdict
from functools import lru_cache

# Local project imports
from src.database import UMLSSession
from src.config import get_config
from resources.common_terms import common_terms
from resources.common_fallbacks import SYMPTOM_NORMALIZATIONS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HIMS-NLP")
HIMS_CONFIG = get_config()

class UMLSMapper:
    """Maps clinical terms to UMLS CUIs with enhanced caching and normalization."""
    
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls) -> 'UMLSMapper':
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.term_cache = LRUCache(maxsize=10000)
        self.symptom_normalizations = self._load_symptom_normalizations()
        self.map_terms_to_cuis_batch(common_terms)
        self._initialized = True
        logger.info("UMLSMapper initialized successfully")
    
    def _load_symptom_normalizations(self) -> Dict[str, str]:
        try:
            return SYMPTOM_NORMALIZATIONS
        except Exception as e:
            logger.error(f"Failed to load symptom normalizations: {e}")
            return {}
    
    def normalize_symptom(self, symptom: str) -> str:
        symptom_lower = symptom.lower()
        return self.symptom_normalizations.get(symptom_lower, symptom_lower)
    
    @lru_cache(maxsize=10000)
    def map_term_to_cui(self, term: str) -> List[str]:
        term = self.normalize_symptom(term)
        if term in self.term_cache:
            return self.term_cache[term]
        
        try:
            with UMLSSession() as session:
                query = text("""
                    SELECT DISTINCT cui
                    FROM umls.mrconso
                    WHERE LOWER(str) = :term
                    AND lat = :language AND suppress = 'N'
                    AND sab IN :trusted_sources
                    LIMIT 1
                """)
                result = session.execute(
                    query,
                    {
                        'term': term,
                        'language': HIMS_CONFIG["UMLS_LANGUAGE"],
                        'trusted_sources': tuple(HIMS_CONFIG["TRUSTED_SOURCES"])
                    }
                ).fetchone()
                
                cui = [result[0]] if result else []
                self.term_cache[term] = cui
                return cui
        except Exception as e:
            logger.error(f"Error mapping term '{term}' to CUI: {e}")
            return []
    
    def map_terms_to_cuis_batch(self, terms: List[str]) -> Dict[str, List[str]]:
        start_time = time.time()
        if not terms:
            return {}
        normalized_terms = [self.normalize_symptom(t) for t in terms]
        
        results = {}
        uncached_terms = []
        
        for term in normalized_terms:
            if term in self.term_cache:
                results[term] = self.term_cache[term]
            else:
                uncached_terms.append(term)
        
        if uncached_terms:
            try:
                with UMLSSession() as session:
                    query = text("""
                        SELECT LOWER(str) AS term_str, cui
                        FROM umls.mrconso
                        WHERE LOWER(str) IN :terms
                        AND lat = :language AND suppress = 'N'
                        AND sab IN :trusted_sources
                    """)
                    db_results = session.execute(
                        query,
                        {
                            'terms': tuple(uncached_terms),
                            'language': HIMS_CONFIG["UMLS_LANGUAGE"],
                            'trusted_sources': tuple(HIMS_CONFIG["TRUSTED_SOURCES"])
                        }
                    ).fetchall()
                    
                    term_map = defaultdict(list)
                    for row in db_results:
                        term_map[row.term_str].append(row.cui)
                    
                    for term in uncached_terms:
                        cuis = term_map.get(term, [])
                        self.term_cache[term] = cuis
                        results[term] = cuis
            except Exception as e:
                logger.error(f"Error in batch UMLS query: {e}")
                for term in uncached_terms:
                    results[term] = self.map_term_to_cui(term)
        
        original_results = {orig_term: results.get(self.normalize_symptom(orig_term), []) 
                           for orig_term in terms}
        
        logger.debug(f"Batch UMLS mapping took {time.time() - start_time:.3f} seconds")
        return original_results