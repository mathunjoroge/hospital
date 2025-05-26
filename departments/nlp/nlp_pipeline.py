import spacy
from spacy.language import Language
from medspacy.target_matcher import TargetMatcher, TargetRule
from medspacy.context import ConText
from scispacy.linking import EntityLinker
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import requests
from departments.nlp.logging_setup import get_logger
from departments.nlp.knowledge_base_io import load_knowledge_base, invalidate_cache
from config import (
    MONGO_URI,
    DB_NAME,
    KB_PREFIX,
    UTS_API_KEY,
    UTS_BASE_URL,
    UTS_AUTH_URL
)

logger = get_logger()
_nlp_pipeline = None


def get_umls_service_ticket():
    try:
        # Step 1: Get TGT
        response = requests.post(UTS_AUTH_URL, data={'apikey': UTS_API_KEY})
        if response.status_code != 201:
            raise Exception(f"Failed to get TGT from UMLS: {response.status_code}, {response.text}")

        # Extract TGT URL from HTML <form action="...">
        import re
        match = re.search(r'action="(.+?)"', response.text)
        if not match:
            raise Exception("Failed to extract TGT URL from response")
        tgt_url = match.group(1)

        # Step 2: Get service ticket
        st_response = requests.post(tgt_url, data={'service': UTS_BASE_URL})
        if st_response.status_code != 200:
            raise Exception(f"Failed to get Service Ticket: {st_response.status_code}, {st_response.text}")

        return st_response.text.strip()
    except Exception as e:
        logger.error(f"Error retrieving UMLS service ticket: {e}")
        return None



def search_umls_cui(term: str):
    try:
        ticket = get_umls_service_ticket()
        if not ticket:
            return None
        params = {
            'string': term,
            'ticket': ticket,
            'searchType': 'exact',
            'sabs': 'SNOMEDCT_US'
        }
        response = requests.get(f"{UTS_BASE_URL}/search/current", params=params)

        if response.status_code != 200:
            logger.warning(f"UMLS API returned status {response.status_code} for term '{term}'")
            return None
        data = response.json()
        results = data.get('result', {}).get('results', [])
        if results:
            cui = results[0].get('ui')
            return cui if cui and cui != 'NONE' else None
    except Exception as e:
        logger.warning(f"Failed to fetch UMLS CUI for '{term}': {e}")
    return None


def get_nlp() -> Language:
    global _nlp_pipeline
    if _nlp_pipeline is not None and "medspacy_target_matcher" in _nlp_pipeline.pipe_names:
        logger.debug("Returning cached NLP pipeline with medspacy_target_matcher")
        return _nlp_pipeline

    try:
        nlp = spacy.load("en_core_sci_sm", disable=["lemmatizer", "parser", "ner"])
        nlp.add_pipe("sentencizer", first=True)

        try:
            nlp.add_pipe("medspacy_pyrush", before="sentencizer")
        except Exception as e:
            logger.warning(f"Failed to add medspacy_pyrush: {str(e)}. Using sentencizer only.")

        if "medspacy_target_matcher" not in nlp.pipe_names:
            nlp.add_pipe("medspacy_target_matcher")
            logger.info("Added medspacy_target_matcher to pipeline")

        try:
            client = MongoClient(MONGO_URI)
            client.admin.command('ping')
            db = client[DB_NAME]
            symptoms_collection = db[f'{KB_PREFIX}symptoms']

            kb = load_knowledge_base()
            logger.debug(f"Knowledge base version: {kb.get('version', 'Unknown')}, last updated: {kb.get('last_updated', 'Unknown')}")

            symptom_rules = []
            valid_count = 0
            invalid_count = 0
            for doc in symptoms_collection.find():
                if 'symptom' not in doc or not doc['symptom'] or len(doc['symptom']) < 2:
                    logger.warning(f"Skipping symptom document {doc.get('_id')}: Invalid symptom name")
                    invalid_count += 1
                    continue

                cui = doc.get('umls_cui', 'Unknown')
                if not cui or cui == 'Unknown' or not isinstance(cui, str) or not cui.startswith('C'):
                    logger.debug(f"Fetching UMLS CUI for symptom: {doc['symptom']}")
                    fetched_cui = search_umls_cui(doc['symptom'])
                    cui = fetched_cui if fetched_cui else 'Unknown'

                attributes = {
                    "cui": cui,
                    "category": doc.get('category', 'Unknown'),
                    "semantic_type": doc.get('semantic_type', 'Unknown'),
                    "icd10": doc.get('icd10', None)
                }

                if attributes['category'] != 'Unknown' and attributes['category'] not in kb.get('symptoms', {}):
                    logger.warning(f"Invalid category for symptom '{doc['symptom']}': {attributes['category']}")

                symptom_rules.append(
                    TargetRule(
                        literal=doc['symptom'].lower(),
                        category="SYMPTOM",
                        attributes=attributes
                    )
                )
                valid_count += 1

            logger.info(f"Processed {valid_count} valid and {invalid_count} invalid symptom documents")
            if symptom_rules:
                target_matcher = nlp.get_pipe("medspacy_target_matcher")
                target_matcher.add(symptom_rules)
                logger.info(f"Loaded {len(symptom_rules)} symptom rules from MongoDB")
            else:
                logger.warning("No valid symptom rules found in MongoDB.")
            client.close()
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}. Invalidating cache.")
            invalidate_cache()

        if "medspacy_context" not in nlp.pipe_names:
            nlp.add_pipe("medspacy_context", after="medspacy_target_matcher")
            logger.info("Added medspacy_context to pipeline")

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
            logger.warning(f"Failed to add scispacy_linker: {str(e)}. Continuing without UMLS linking.")

        nlp.initialize()
        logger.info(f"Initialized medspacy pipeline with components: {nlp.pipe_names}")

        _nlp_pipeline = nlp
        return nlp

    except Exception as e:
        logger.error(f"Failed to initialize NLP pipeline: {str(e)}")
        nlp = spacy.load("en_core_sci_sm", disable=["lemmatizer", "parser", "ner"])
        nlp.add_pipe("sentencizer", first=True)
        logger.warning("Using fallback spacy pipeline without parser, ner, or linker")
        _nlp_pipeline = nlp
        return nlp
