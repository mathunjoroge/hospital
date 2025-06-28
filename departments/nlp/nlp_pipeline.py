import logging
import spacy
from spacy.tokens import Token, Span
from spacy.language import Language
from packaging.version import Version, InvalidVersion
from packaging.specifiers import SpecifierSet
from medspacy.target_matcher import TargetMatcher, TargetRule
from medspacy.context import ConText, ConTextRule
from scispacy.linking import EntityLinker
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from departments.models.pharmacy import Batch
from departments.nlp.logging_setup import get_logger
from departments.nlp.nlp_utils import preprocess_text, FALLBACK_CUI_MAP
from departments.nlp.nlp_common import clean_term, STOP_TERMS
import pkg_resources
import os
import time
import pickle
from cachetools import LRUCache
import re
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import (
    MONGO_URI,
    DB_NAME,
    SYMPTOMS_COLLECTION,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    CACHE_DIR,
    BATCH_SIZE
)

logger = get_logger(__name__)

_nlp_pipeline = None
_sci_ner = None
_cui_cache = LRUCache(maxsize=10000)
_phrase_cache = LRUCache(maxsize=1000)
_cache_file = os.path.join(CACHE_DIR or 'data_cache', "cui_cache.pkl")

class SciBERTWrapper:
    def __init__(self, model_name="en_core_sci_sm", disable_linker=True):
        try:
            self.nlp = spacy.load(model_name, disable=["lemmatizer"])
            logger.info(f"Loaded SpaCy model: {model_name}")
            if disable_linker and "entity_linker" in self.nlp.pipe_names:
                self.nlp.remove_pipe("entity_linker")
                logger.info("Removed entity_linker to avoid nmslib dependency.")
        except OSError as e:
            logger.error(f"Failed to load spaCy model {model_name}: {e}. Using blank model.")
            self.nlp = spacy.blank("en")

    def extract_entities(self, text):
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def analyze(self, text):
        doc = self.nlp(text)
        return {
            "tokens": [(token.text, token.pos_, token.dep_) for token in doc],
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "sentences": [sent.text for sent in doc.sents],
        }

# Initialize _sci_ner
try:
    _sci_ner = SciBERTWrapper(model_name="en_core_sci_sm", disable_linker=True)
    logger.info("Initialized _sci_ner with en_core_sci_sm")
except Exception as e:
    logger.error(f"Failed to initialize _sci_ner: {e}")
    _sci_ner = SciBERTWrapper(model_name="en", disable_linker=True)

# Initialize PostgreSQL connection pool
try:
    pool = SimpleConnectionPool(
        minconn=5,
        maxconn=20,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        cursor_factory=RealDictCursor,
        connect_timeout=10
    )
    logger.info("Initialized PostgreSQL connection pool")
except Exception as e:
    logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
    pool = None

@contextmanager
def get_postgres_connection(readonly=True):
    if pool is None:
        logger.error("Connection pool not initialized. Check PostgreSQL configuration.")
        yield None
    else:
        conn = None
        cursor = None
        try:
            conn = pool.getconn()
            conn.set_session(autocommit=False, readonly=readonly)
            cursor = conn.cursor()
            logger.debug(f"Connected to PostgreSQL database from pool (readonly={readonly})")
            yield cursor
        except Exception as e:
            logger.error(f"Failed to get connection or cursor from pool: {e}")
            yield None
        finally:
            if cursor:
                cursor.close()
                logger.debug("Closed PostgreSQL cursor")
            if conn:
                if not readonly:
                    try:
                        conn.commit()
                    except Exception as e:
                        logger.error(f"Failed to commit transaction: {e}")
                        conn.rollback()
                pool.putconn(conn)
                logger.debug("Returned PostgreSQL connection to pool")

def load_cui_cache():
    global _cui_cache
    from departments.nlp.knowledge_base_io import load_knowledge_base
    knowledge_base = load_knowledge_base() or {}
    if os.path.exists(_cache_file):
        try:
            start_time = time.time()
            with open(_cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                cache_version = cached_data.get('version', 'Unknown')
                umls_version = knowledge_base.get('umls_version', 'Unknown')
                if cache_version == umls_version:
                    _cui_cache.update(cached_data.get('data', {}))
                    logger.info(f"Loaded CUI cache for UMLS version {umls_version} with {len(_cui_cache)} entries")
                else:
                    logger.warning(f"Cache version mismatch: {cache_version} != {umls_version}. Clearing cache.")
                    _cui_cache = LRUCache(maxsize=10000)
            logger.info(f"Loaded CUI cache, took {time.time() - start_time:.3f} seconds")
        except Exception as e:
            logger.warning(f"Failed to load CUI cache: {e}")
            _cui_cache = LRUCache(maxsize=10000)

def save_cui_cache():
    from departments.nlp.knowledge_base_io import load_knowledge_base
    knowledge_base = load_knowledge_base() or {}
    try:
        os.makedirs(os.path.dirname(_cache_file), exist_ok=True)
        start_time = time.time()
        umls_version = knowledge_base.get('umls_version', 'Unknown')
        with open(_cache_file, 'wb') as f:
            pickle.dump({'version': umls_version, 'data': dict(_cui_cache)}, f)
        logger.debug(f"Saved CUI cache to {_cache_file}, took {time.time() - start_time:.3f} seconds")
    except Exception as e:
        logger.warning(f"Failed to save CUI cache: {e}")

@Language.component("custom_umls_linker")
def custom_umls_linker(doc):
    terms = [clean_term(ent.text) for ent in doc.ents if clean_term(ent.text)]
    cui_results = search_local_umls_cui(terms)
    for ent in doc.ents:
        cleaned = clean_term(ent.text)
        if cleaned in cui_results and cui_results[cleaned]:
            ent._.umls_cui = cui_results[cleaned]
    return doc

def search_local_umls_cui(terms: list, max_attempts=3, batch_size=100, max_tsquery_bytes=500000):
    from departments.nlp.knowledge_base_io import load_knowledge_base
    knowledge_base = load_knowledge_base() or {}
    stop_terms = knowledge_base.get('medical_stop_words', set())
    stop_terms.update({
        'the', 'and', 'or', 'is', 'started', 'days', 'weeks', 'months', 'years', 'ago',
        'taking', 'makes', 'alleviates', 'foods', 'fats', 'strong', 'tea', 'licking', 'salt', 'worse'
    })
    start_time = time.time()
    cleaned_terms = [clean_term(term) for term in terms if clean_term(term)]
    results = {term: None for term in cleaned_terms}

    # Cache and fallback lookup
    for term in cleaned_terms:
        if term in _cui_cache:
            results[term] = _cui_cache[term]
            logger.debug(f"Cache hit: {term} -> {_cui_cache[term]}")
        elif term in FALLBACK_CUI_MAP:
            results[term] = FALLBACK_CUI_MAP[term]['umls_cui']
            _cui_cache[term] = results[term]
            logger.debug(f"Fallback hit: {term} -> {FALLBACK_CUI_MAP[term]['umls_cui']}")

    terms_to_query = [t for t in cleaned_terms if results[t] is None and t and len(t) <= 100 and t not in stop_terms]
    if not terms_to_query:
        logger.info(f"No terms to query, took {time.time() - start_time:.3f} seconds")
        return results

    with get_postgres_connection() as cursor:
        if not cursor:
            logger.error("No database cursor available")
            if _nlp_pipeline and "scispacy_linker" in _nlp_pipeline.pipe_names:
                logger.info("Falling back to scispacy_linker for unmapped terms")
                linker = _nlp_pipeline.get_pipe("scispacy_linker")
                for term in terms_to_query:
                    doc = _nlp_pipeline(term)
                    for ent in doc.ents:
                        if ent._.kb_ents:
                            cui = ent._.kb_ents[0][0]
                            results[term] = cui
                            _cui_cache[term] = cui
            return results

        for attempt in range(max_attempts):
            try:
                # Exact match query
                query_start = time.time()
                date_pattern = r'^(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\w{3}-\d{4})\s*'
                query = """
                    SELECT DISTINCT
                        CASE
                            WHEN c.STR ~ %s THEN REGEXP_REPLACE(c.STR, %s, '')
                            ELSE c.STR
                        END AS cleaned_str,
                        c.CUI,
                        r.MRRANK_RANK
                    FROM umls.MRCONSO c
                    LEFT JOIN umls.MRRANK r ON c.SAB = r.SAB AND c.TTY = r.TTY
                    WHERE c.SAB IN ('SNOMEDCT_US', 'MSH', 'RXNORM')
                    AND LOWER(c.STR) IN %s
                    ORDER BY r.MRRANK_RANK DESC NULLS LAST
                """
                cursor.execute(query, (date_pattern, date_pattern, tuple(terms_to_query)))
                for row in cursor.fetchall():
                    term = clean_term(row['cleaned_str'])
                    if term:
                        results[term] = row['cui']
                        _cui_cache[term] = row['cui']
                logger.debug(f"Exact match query for {len(terms_to_query)} terms took {time.time() - query_start:.2f} seconds")

                # Full-text search
                remaining = [t for t in terms_to_query if results[t] is None]
                for i in range(0, len(remaining), batch_size):
                    batch_terms = remaining[i:i + batch_size]
                    tsquery_parts = [" & ".join(t.split()) for t in batch_terms]
                    tsquery = ' & '.join(f'({part}:*)' for part in tsquery_parts)
                    if not tsquery or len(tsquery.encode()) > max_tsquery_bytes:
                        logger.warning(f"Skipping oversized tsquery for batch {i//batch_size}")
                        continue
                    query = """
                        SELECT DISTINCT
                            CASE
                                WHEN c.STR ~ %s THEN REGEXP_REPLACE(c.STR, %s, '')
                                ELSE c.STR
                            END AS cleaned_str,
                            c.CUI,
                            ts_rank(to_tsvector('english', c.STR), to_tsquery('english', %s)) AS rank
                        FROM umls.MRCONSO c
                        WHERE c.SAB IN ('SNOMEDCT_US', 'MSH', 'RXNORM')
                        AND to_tsvector('english', c.STR) @@ to_tsquery('english', %s)
                        AND octet_length(c.STR) <= 1048575
                        ORDER BY rank DESC
                        LIMIT 1
                    """
                    cursor.execute(query, (date_pattern, date_pattern, tsquery, tsquery))
                    for row in cursor.fetchall():
                        term = clean_term(row['cleaned_str'])
                        if term:
                            results[term] = row['cui']
                            _cui_cache[term] = row['cui']
                    logger.debug(f"Full-text search for {len(batch_terms)} terms took {time.time() - query_start:.3f} seconds")
                cursor.connection.commit()
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                cursor.connection.rollback()
                if attempt == max_attempts - 1:
                    logger.error("Max retries reached")
                    return results
        save_cui_cache()
        logger.info(f"Processed {len(terms)} terms, took {time.time() - start_time:.3f} seconds")
        return results

def infer_category(cui):
    from departments.nlp.knowledge_base_io import REQUIRED_CATEGORIES
    with get_postgres_connection() as cursor:
        if not cursor:
            logger.error("No database cursor available")
            return 'unknown'
        try:
            query = """
                SELECT DISTINCT m.STR
                FROM umls.MRREL r
                JOIN umls.MRCONSO m ON r.CUI2 = m.CUI
                WHERE r.CUI1 = %s AND r.REL = 'PAR' AND r.SAB = 'SNOMEDCT_US'
                LIMIT 1
            """
            cursor.execute(query, (cui,))
            result = cursor.fetchone()
            category = result['str'].lower() if result else 'unknown'
            if category in REQUIRED_CATEGORIES:
                return 'pulmonary' if category == 'respiratory' else category
            return 'unknown'
        except Exception as e:
            logger.error(f"Failed to infer category for CUI {cui}: {e}")
            return 'unknown'

def get_semantic_types(cuis: list) -> dict:
    start_time = time.time()
    results = {cui: [] for cui in cuis}
    for cui in cuis:
        for term, data in FALLBACK_CUI_MAP.items():
            if data['umls_cui'] == cui:
                results[cui].append(data['semantic_type'])
                logger.debug(f"Fallback semantic type: {cui} -> {data['semantic_type']}")

    cuis_to_query = [cui for cui in set(cuis) if not results[cui] and cui and isinstance(cui, str) and cui.startswith('C')]
    if not cuis_to_query:
        logger.debug(f"No CUIs to query, took {time.time() - start_time:.3f} seconds")
        return {k: v[0] if v else 'unknown' for k, v in results.items()}

    with get_postgres_connection() as cursor:
        if not cursor:
            logger.error("No database cursor available")
            return {k: v[0] if v else 'unknown' for k, v in results.items()}
        try:
            query = """
                SELECT sty.CUI, array_agg(sty.STY) AS semantic_types
                FROM umls.MRSTY sty
                WHERE sty.CUI IN %s
                GROUP BY sty.CUI
            """
            cursor.execute(query, (tuple(cuis_to_query),))
            for row in cursor.fetchall():
                results[row['cui']] = row['semantic_types'] or ['unknown']
            logger.debug(f"Semantic type query for {len(cuis_to_query)} CUIs took {time.time() - start_time:.3f} seconds")
            return {k: v[0] if len(v) == 1 else v for k, v in results.items()}
        except Exception as e:
            logger.error(f"Failed to query semantic types: {e}")
            return {k: v[0] if v else 'unknown' for k, v in results.items()}

def process_text_batch(batch):
    try:
        return list(zip(batch, _sci_ner.nlp.pipe(batch)))
    except Exception as e:
        logger.error(f"Failed to process text batch: {e}")
        return [(text, None) for text in batch]

def extract_clinical_phrases(texts):
    from departments.nlp.knowledge_base_io import load_knowledge_base
    knowledge_base = load_knowledge_base() or {}
    stop_terms = set(knowledge_base.get('medical_stop_words', set())) | STOP_TERMS
    start_time = time.time()
    if not isinstance(texts, list):
        texts = [texts]
    cached_results = [(_phrase_cache[text] if text in _phrase_cache else None) for text in texts]
    if all(r is not None for r in cached_results):
        logger.debug(f"All {len(texts)} texts found in phrase cache, took {time.time() - start_time:.3f} seconds")
        return cached_results if len(texts) > 1 else cached_results[0] or []

    uncached_texts = [t for t, r in zip(texts, cached_results) if r is None]
    results = [[] if r is None else r for r in cached_results]
    batch_size = BATCH_SIZE
    with ThreadPoolExecutor(max_workers=4) as executor:
        batches = [uncached_texts[i:i + batch_size] for i in range(0, len(uncached_texts), batch_size)]
        for i, batch_result in enumerate(executor.map(process_text_batch, batches)):
            for text, doc in batch_result:
                if doc is None:
                    logger.warning(f"Failed to process text: {text[:50]}...")
                    results[texts.index(text)] = []
                    continue
                phrases = []
                for chunk in doc.noun_chunks:
                    cleaned = clean_term(chunk.text)
                    if cleaned and len(cleaned) <= 50 and len(cleaned.split()) <= 5 and cleaned not in stop_terms:
                        phrases.append(cleaned)
                for ent in doc.ents:
                    if len(ent.text.strip()) <= 2:
                        continue
                    cleaned = clean_term(ent.text)
                    if cleaned and len(cleaned) <= 50 and len(cleaned.split()) <= 5 and cleaned not in stop_terms:
                        phrases.append(cleaned)
                    if ' and ' in ent.text:
                        sub_phrases = [clean_term(p) for p in ent.text.split(' and ')]
                        phrases.extend(p for p in sub_phrases if p and len(p) <= 50 and len(p.split()) <= 5 and p not in stop_terms)
                for sent in doc.sents:
                    if len(sent.text.strip().split()) > 5:
                        for sub_ent in sent.ents:
                            cleaned = clean_term(sub_ent.text)
                            if cleaned and len(cleaned) <= 50 and len(cleaned.split()) <= 5 and cleaned not in stop_terms:
                                phrases.append(cleaned)
                filtered_phrases = list(set(phrases))[:50]
                _phrase_cache[text] = filtered_phrases
                try:
                    results[texts.index(text)] = filtered_phrases
                except ValueError:
                    logger.error(f"Text {text[:50]}... not found in original texts list")
                    continue
    logger.info(f"Processed {len(texts)} texts, returned {sum(len(r) for r in results if r is not None)} terms, took {time.time() - start_time:.3f} seconds")
    return results if len(texts) > 1 else results[0] or []

def extract_aggravating_alleviating(text: str, factor_type: str) -> str:
    if not isinstance(text, str) or not text.strip():
        logger.debug(f"Invalid text for {factor_type} factor extraction: {text}")
        return "unknown"
    if factor_type not in ["aggravating", "alleviating"]:
        logger.error(f"Invalid factor_type: {factor_type}")
        return "unknown"

    text_clean = preprocess_text(text).lower()
    text_clean = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', text_clean)).strip()
    if not text_clean:
        logger.debug(f"Empty text after preprocessing for {factor_type} factor")
        return "unknown"

    aggravating_patterns = [
        r'\b(makes\s+worse|worsens|aggravates|triggers|exacerbates)\s+([\w\s-]+?)(?=\s*|,\s*|\s+and\b|\s+or\b|$)',
        r'\b(worse\s+with|caused\s+by|triggered\s+by)\s+([\w\s-]+?)(?=\s*|,\s*|\s+and\b|\s+or\b|$)',
        r'\b(not\s+better\s+with|increases\s+with)\s+([\w\s-]+?)(?=\s*|,\s*|\s+and\b|\s+or\b|$)'
    ]
    alleviating_patterns = [
        r'\b(alleviates|relieves|improves|better\s+with|reduces|eases)\s+([\w\s-]+?)(?=\s*|,\s*|\s+and\b|\s+or\b|$)',
        r'\b(reduced\s+by|eased\s+by|improved\s+by)\s+([\w\s-]+?)(?=\s*|,\s*|\s+and\b|\s+or\b|$)',
        r'\b(not\s+worse\s+with|decreases\s+with)\s+([\w\s-]+?)(?=\s*|,\s*|\s+and\b|\s+or\b|$)'
    ]

    patterns = aggravating_patterns if factor_type == "aggravating" else alleviating_patterns
    for pattern in patterns:
        try:
            logger.debug(f"Compiling regex pattern: {pattern}")
            match = re.search(pattern, text_clean, re.IGNORECASE)
            if match:
                factor = match.group(2).strip()
                factor_clean = clean_term(factor)
                if factor_clean:
                    logger.debug(f"Extracted {factor_type} factor: '{factor_clean}'")
                    return factor_clean
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {str(e)}")
            continue
    logger.debug(f"No {factor_type} factor found in text: {text_clean[:50]}...")
    return "unknown"

def generate_symptom_rules(symptoms_collection):
    from departments.nlp.knowledge_base_io import REQUIRED_CATEGORIES
    rules = []
    for doc in symptoms_collection.find():
        if 'symptom' in doc and doc['symptom'] and 'umls_cui' in doc:
            literal = clean_term(doc['symptom'])
            category = doc.get('category', 'unknown')
            if category not in REQUIRED_CATEGORIES:
                logger.warning(f"Invalid category '{category}' for symptom '{literal}', setting to 'unknown'")
                category = 'unknown'
            if literal:
                rules.append(TargetRule(
                    literal=literal,
                    category="SYMPTOM",
                    attributes={
                        "umls_cui": doc['umls_cui'],
                        "category": category,
                        "semantic_type": doc.get('semantic_type', 'unknown'),
                        "icd10": doc.get('icd10', None)
                    }
                ))
    logger.info(f"Generated {len(rules)} symptom rules from MongoDB")
    return rules

def process_symptom_batch(symptom_docs: list, symptom_rules: list, counts: dict, kb: dict, symptoms_collection):
    from departments.nlp.knowledge_base_io import REQUIRED_CATEGORIES
    if not symptom_docs:
        logger.warning("No symptom documents to process")
        return
    symptom_texts = [doc['symptom'] for doc in symptom_docs if 'symptom' in doc and doc['symptom'] and len(doc['symptom']) >= 2]
    terms_list = extract_clinical_phrases(symptom_texts)
    if terms_list is None:
        logger.error("extract_clinical_phrases returned None")
        counts['invalid'] += len(symptom_docs)
        return
    logger.debug(f"Extracted {sum(len(terms) for terms in terms_list if terms is not None)} terms from {len(symptom_texts)} symptoms")
    cui_results = search_local_umls_cui([term for terms in terms_list if terms is not None for term in terms])

    all_cuis = []
    for terms in terms_list:
        if terms is None:
            continue
        for term in terms:
            cui = cui_results.get(clean_term(term))
            if cui and cui.startswith('C'):
                all_cuis.append(cui)
    semantic_types = get_semantic_types(list(set(all_cuis)))

    updates = []
    for doc, terms in zip(symptom_docs, terms_list):
        if 'symptom' not in doc or not doc['symptom'] or len(doc['symptom']) < 2:
            counts['invalid'] += 1
            continue

        cui = doc.get('umls_cui', 'unknown')
        if not cui or cui == 'unknown' or not isinstance(cui, str) or not cui.startswith('C'):
            fetched_cui = None
            if terms is not None:
                for term in terms:
                    fetched_cui = cui_results.get(clean_term(term))
                    if fetched_cui:
                        updates.append({
                            'filter': {'_id': doc['_id']},
                            'update': {'$set': {'umls_cui': fetched_cui}}
                        })
                        break
            cui = fetched_cui if fetched_cui else 'NO_CUI_' + clean_term(doc['symptom'])

        semantic_type = semantic_types.get(cui, 'unknown')
        category = doc.get('category', infer_category(cui))
        if category not in REQUIRED_CATEGORIES:
            logger.warning(f"Invalid category '{category}' for symptom '{doc['symptom']}', setting to 'unknown'")
            category = 'unknown'

        attributes = {
            "umls_cui": cui,
            "category": category,
            "semantic_type": semantic_type,
            "icd10": doc.get('icd10', None)
        }

        if attributes['category'] != 'unknown' and attributes['category'] not in kb.get('clinical_pathways', {}):
            logger.warning(f"Category '{attributes['category']}' not in clinical_pathways, setting to 'unknown'")
            attributes['category'] = 'unknown'

        literal = clean_term(doc['symptom'])
        if not literal:
            counts['invalid'] += 1
            continue

        symptom_rules.append(
            TargetRule(
                literal=literal,
                category="SYMPTOM",
                attributes=attributes
            )
        )
        counts['valid'] += 1

    if updates:
        try:
            symptoms_collection.bulk_write([UpdateOne(u['filter'], u['update']) for u in updates], ordered=False)
            logger.info(f"Updated {len(updates)} MongoDB documents in batch")
        except Exception as e:
            logger.error(f"Failed to update MongoDB: {str(e)}")

def get_nlp(max_attempts=3) -> Language:
    global _nlp_pipeline
    start_time = time.time()
    if _nlp_pipeline is not None:
        logger.debug(f"Returning cached NLP pipeline with components: {_nlp_pipeline.pipe_names}, took {time.time() - start_time:.3f} seconds")
        return _nlp_pipeline

    for attempt in range(1, max_attempts + 1):
        try:
            # Check if required packages are installed (no version checks)
            required_packages = ['spacy', 'medspacy', 'scispacy']
            for pkg in required_packages:
                try:
                    pkg_resources.get_distribution(pkg)
                    logger.debug(f"{pkg} is installed")
                except pkg_resources.DistributionNotFound:
                    logger.error(f"{pkg} not installed")
                    raise ImportError(f"{pkg} is required")

            # Load spaCy model
            try:
                nlp = spacy.load("en_core_sci_sm", disable=["lemmatizer"])
                logger.info(f"Loaded base spaCy model: en_core_sci_sm")
            except OSError as e:
                logger.error(f"Failed to load en_core_sci_sm: {e}. Falling back to blank English model.")
                nlp = spacy.blank("en")
                logger.info("Initialized blank English model")

            # Register spaCy extensions
            for ext, default in [
                ('umls_cui', None),
                ('category', None),
                ('semantic_type', None),
                ('icd10', None)
            ]:
                if not Span.has_extension(ext):
                    Span.set_extension(ext, default=default)
                if not Token.has_extension(ext):
                    Token.set_extension(ext, default=default)

            nlp.add_pipe("sentencizer", first=True)
            logger.info("Added sentencizer to pipeline")

            # Add symptom matcher
            if "symptom_matcher" not in nlp.pipe_names:
                nlp.add_pipe("medspacy_target_matcher", name="symptom_matcher")
                logger.info("Added medspacy_target_matcher as symptom_matcher to pipeline")
            else:
                logger.debug("symptom_matcher already exists in pipeline, skipping add")

            # Add context component
            if "medspacy_context" not in nlp.pipe_names:
                context = ConText(nlp, rules="default")
                context.add([
                    ConTextRule(literal="no|without|denies|not", category="NEGATED", pattern=r"\b(no|without|denies|not)\b", direction="FORWARD"),
                    ConTextRule(literal="absence of", category="NEGATED", pattern=r"\b(absence\s+of)\b", direction="FORWARD"),
                    ConTextRule(literal="negative for", category="NEGATED", pattern=r"\b(negative\s+for)\b", direction="FORWARD"),
                    ConTextRule(literal="no evidence of", category="NEGATED", pattern=r"\b(no\s+evidence\s+of)\b", direction="FORWARD")
                ])
                nlp.add_pipe("medspacy_context", after="symptom_matcher")
                logger.info("Added medspacy_context with custom negation rules")
            else:
                logger.debug("medspacy_context already exists in pipeline, skipping add")

            # Add scispacy linker or custom fallback
            try:
                if "scispacy_linker" not in nlp.pipe_names:
                    nlp.add_pipe("scispacy_linker", config={
                        "resolve_abbreviations": True,
                        "linker_name": "umls",
                        "k": 10,
                        "threshold": 0.7
                    })
                    logger.info("Added scispacy_linker to pipeline")
            except Exception as e:
                logger.warning(f"Failed to add scispacy_linker: {e}. Using custom UMLS linker.")
                if "custom_umls_linker" not in nlp.pipe_names:
                    nlp.add_pipe("custom_umls_linker")
                    logger.info("Added custom_umls_linker to pipeline")

            # MongoDB operations
            client = None
            try:
                client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
                client.admin.command('ping')
                logger.info("MongoDB connection successful")
                db = client[DB_NAME]
                symptoms_collection = db[SYMPTOMS_COLLECTION]
                diagnosis_treatments_collection = db['diagnosis_treatments']

                # Check if diagnosis_treatments is empty
                if diagnosis_treatments_collection.count_documents({}) == 0:
                    logger.warning("MongoDB collection diagnosis_treatments is empty. Initializing with placeholder data.")
                    diagnosis_treatments_collection.insert_one({"placeholder": "default"})
                    logger.info("Inserted placeholder document into diagnosis_treatments")

                # Ensure unique index
                index_name = "symptom_cui_unique"
                existing_indexes = symptoms_collection.index_information()
                if index_name not in existing_indexes:
                    null_count = symptoms_collection.count_documents({"symptom": None})
                    if null_count > 0:
                        logger.warning(f"Found {null_count} documents with null 'symptom'. Reviewing for retention.")
                        for doc in symptoms_collection.find({"symptom": None}):
                            if 'category' in doc and doc['category']:
                                symptoms_collection.update_one({'_id': doc['_id']}, {'$set': {'symptom': 'unknown'}})
                            else:
                                symptoms_collection.delete_one({'_id': doc['_id']})
                    symptoms_collection.create_index([('symptom', 1), ('umls_cui', 1)], unique=True, name=index_name)
                    logger.info("Created MongoDB unique index on symptom and umls_cui")
                elif not existing_indexes[index_name].get("unique"):
                    symptoms_collection.drop_index(index_name)
                    symptoms_collection.create_index([('symptom', 1), ('umls_cui', 1)], unique=True, name=index_name)
                    logger.info("Recreated unique index")

                # Generate symptom rules dynamically
                symptom_rules = generate_symptom_rules(symptoms_collection)
                if symptom_rules:
                    symptom_matcher = nlp.get_pipe("symptom_matcher")
                    symptom_matcher.add(symptom_rules)
                    logger.info(f"Added {len(symptom_rules)} symptom rules to matcher")
                else:
                    logger.warning("No valid symptom rules found")

                # Process MongoDB symptoms
                counts = {"valid": 0, "invalid": 0}
                load_cui_cache()
                cursor = symptoms_collection.find().batch_size(100)
                symptom_data = []
                try:
                    from departments.nlp.knowledge_base_io import load_knowledge_base
                    knowledge_base = load_knowledge_base() or {}
                    for doc_data in cursor:
                        symptom_data.append(doc_data)
                        if len(symptom_data) >= 1000:
                            process_symptom_batch(symptom_data, symptom_rules, counts, knowledge_base, symptoms_collection)
                            symptom_data = []
                    if symptom_data:
                        process_symptom_batch(symptom_data, symptom_rules, counts, knowledge_base, symptoms_collection)
                finally:
                    cursor.close()
                logger.info(f"Processed {counts['valid']} valid and {counts['invalid']} invalid symptom documents")

            except ConnectionFailure as e:
                logger.error(f"Failed to connect to MongoDB: {str(e)}")
                from departments.nlp.knowledge_base_io import invalidate_cache
                invalidate_cache()
                raise
            finally:
                if client:
                    client.close()
                    logger.debug("MongoDB client closed")

            # Initialize tokenizer for blank model
            if hasattr(nlp, 'is_blank') and nlp.is_blank and hasattr(nlp, 'initialize'):
                try:
                    nlp.tokenizer.initialize()
                    logger.info("Initialized tokenizer for blank model")
                except Exception as e:
                    logger.warning(f"Couldn't initialize blank model tokenizer: {e}")

            _nlp_pipeline = nlp
            logger.info(f"Pipeline ready with components: {nlp.pipe_names}, total time: {time.time() - start_time:.3f} seconds")
            return nlp

        except Exception as e:
            logger.error(f"Attempt {attempt}/{max_attempts} failed to initialize NLP pipeline: {str(e)}", exc_info=True)
            if attempt == max_attempts:
                logger.error("Max retries reached for NLP pipeline initialization")
                raise
            continue