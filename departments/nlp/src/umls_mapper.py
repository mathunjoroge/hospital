import logging
import time
from typing import List, Dict
from sqlalchemy.sql import text
from cachetools import LRUCache
from threading import Lock
from collections import defaultdict

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
    """
    Maps clinical terms to UMLS CUIs with a unified cache and comprehensive lookups.
    
    This optimized version uses a single, shared LRUCache for both single and
    batch lookups, and retrieves all possible CUIs for a term instead of just one.
    """
    
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls) -> 'UMLSMapper':
        # Slightly more concise singleton pattern
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # Use a single, instance-level cache. This is essential for the batch
        # function to correctly identify which terms are already cached.
        self.term_cache: LRUCache[str, List[str]] = LRUCache(maxsize=10000)
        
        # Directly assign constants instead of using a wrapper method.
        self.symptom_normalizations = SYMPTOM_NORMALIZATIONS
        
        # Pre-warm the cache with common terms on initialization.
        self.map_terms_to_cuis_batch(common_terms)
        self._initialized = True
        logger.info("UMLSMapper initialized with a unified cache.")
    
    def normalize_symptom(self, symptom: str) -> str:
        """Normalizes a symptom string to its canonical form."""
        symptom_lower = symptom.lower().strip()
        return self.symptom_normalizations.get(symptom_lower, symptom_lower)
    
    def map_term_to_cui(self, term: str) -> List[str]:
        """
        Maps a single normalized term to a list of CUIs.
        
        This method no longer uses @lru_cache, relying on the shared self.term_cache.
        It now fetches ALL matching CUIs, not just the first one.
        """
        normalized_term = self.normalize_symptom(term)
        
        # Check the single, shared cache first.
        if normalized_term in self.term_cache:
            return self.term_cache[normalized_term]
        
        try:
            with UMLSSession() as session:
                query = text("""
                    SELECT DISTINCT cui
                    FROM umls.mrconso
                    WHERE LOWER(str) = :term
                    AND lat = :language AND suppress = 'N'
                    AND sab IN :trusted_sources
                """)
                results = session.execute(
                    query,
                    {
                        'term': normalized_term,
                        'language': HIMS_CONFIG["UMLS_LANGUAGE"],
                        'trusted_sources': tuple(HIMS_CONFIG["TRUSTED_SOURCES"])
                    }
                ).fetchall()
                
                # Retrieve all CUIs, not just one.
                cuis = [row[0] for row in results]
                self.term_cache[normalized_term] = cuis
                return cuis
        except Exception as e:
            logger.error(f"Error mapping term '{normalized_term}' to CUI: {e}")
            # Cache the failure to prevent repeated failed lookups for the same term.
            self.term_cache[normalized_term] = []
            return []
    
    def map_terms_to_cuis_batch(self, terms: List[str]) -> Dict[str, List[str]]:
        """Efficiently maps a batch of terms to their CUIs using a single query."""
        start_time = time.time()
        if not terms:
            return {}

        results: Dict[str, List[str]] = {}
        uncached_normalized_terms = set()
        
        # Create a map from original terms to their normalized form.
        norm_map = {orig: self.normalize_symptom(orig) for orig in terms}

        # First pass: check the cache for all unique normalized terms.
        for normalized_term in set(norm_map.values()):
            if normalized_term in self.term_cache:
                results[normalized_term] = self.term_cache[normalized_term]
            else:
                uncached_normalized_terms.add(normalized_term)
        
        # Second pass: query the database for all uncached terms in one go.
        if uncached_normalized_terms:
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
                            'terms': tuple(uncached_normalized_terms),
                            'language': HIMS_CONFIG["UMLS_LANGUAGE"],
                            'trusted_sources': tuple(HIMS_CONFIG["TRUSTED_SOURCES"])
                        }
                    ).fetchall()
                    
                    term_map = defaultdict(list)
                    for row in db_results:
                        term_map[row.term_str].append(row.cui)
                    
                    # Populate cache and results for terms found in the DB.
                    for term in uncached_normalized_terms:
                        cuis = term_map.get(term, [])
                        self.term_cache[term] = cuis
                        results[term] = cuis
            except Exception as e:
                logger.error(f"Error in batch UMLS query: {e}. Falling back to single queries.")
                # Fallback: if batch query fails, process one-by-one.
                for term in uncached_normalized_terms:
                    results[term] = self.map_term_to_cui(term)
        
        # Map the results from normalized terms back to the original input terms.
        final_results = {orig_term: results.get(norm_term, []) for orig_term, norm_term in norm_map.items()}
        
        duration = time.time() - start_time
        logger.debug(f"Batch UMLS mapping for {len(terms)} terms took {duration:.3f} seconds.")
        return final_results