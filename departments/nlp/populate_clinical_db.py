import os
import json
import random
import time
import urllib.parse
import asyncio
import sys
import re
import hashlib
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, PyMongoError
import logging
from tqdm import tqdm
from fuzzywuzzy import fuzz, process
import string
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import multiprocessing
import yaml
from urllib.robotparser import RobotFileParser

# Constants
CACHE_TTL = 86400  # 24 hours
REQUEST_TIMEOUT = 45
BASE_SCRAPE_DELAY = 5
MAX_SYMPTOMS_PER_SOURCE = 15
BATCH_SIZE = 10
MIN_SYMPTOMS = 2
MAX_RETRIES = 3
MAX_CACHE_SIZE = 1000
MIN_DELAY = 1
MAX_DELAY = 30
CHECKPOINT_FILE = 'checkpoint.json'
CIRCUIT_BREAKER_THRESHOLD = 10
CIRCUIT_RESET_TIME = 300  # 5 minutes
ONTOLOGY_FILE = 'medical_ontology.txt'
CDC_API_BASE = 'https://api.cdc.gov/health/{condition}/symptoms'

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clinical_db_population.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# User Agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1'
]

# CDC Conditions
CDC_CONDITIONS = [
    'malaria', 'tuberculosis', 'hiv/aids', 'influenza', 'dengue fever',
    'zika virus', 'cholera', 'alcohol poisoning', 'alcoholic hepatitis',
    'lyme disease', 'hepatitis b', 'hepatitis c', 'covid-19'
]

def load_url_mappings() -> Dict[str, Dict[str, str]]:
    mappings_file = Path('url_mappings.yaml')
    default_mappings = {
        'mayo_clinic': {
            'atrial fibrillation': 'atrial-fibrillation/symptoms-causes/syc-20350624',
            # ... other mappings
        },
        'webmd': {
            'atrial fibrillation': 'heart-disease/atrial-fibrillation',
            # ... other mappings
        },
        'medlineplus': {
            'atrial fibrillation': 'atrialfibrillation.html',
            # ... other mappings
        }
    }
    
    if mappings_file.exists():
        with mappings_file.open('r') as f:
            return yaml.safe_load(f) or default_mappings
    else:
        with mappings_file.open('w') as f:
            yaml.safe_dump(default_mappings, f)
        return default_mappings

URL_MAPPINGS = load_url_mappings()

class Config:
    def __init__(self):
        load_dotenv()
        self.mongo_uri = os.getenv('MONGO_URI')
        if not self.mongo_uri:
            raise ValueError("MONGO_URI environment variable is required")
        self.db_name = os.getenv('DB_NAME', 'clinical_db')
        self.kb_prefix = os.getenv('KB_PREFIX', 'kb_')
        self.cache_dir = Path(os.getenv('CACHE_DIR', 'data_cache'))
        self.backup_dir = Path(os.getenv('BACKUP_DIR', 'backups'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.dry_run = os.getenv('DRY_RUN', 'false').lower() == 'true'

class ClinicalKnowledgePopulator:
    def __init__(self, config: Config):
        self.config = config
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.failed_fetches = []
        self.executor = ThreadPoolExecutor(max_workers=max(4, multiprocessing.cpu_count()))
        self.medical_terms = self._load_ontology()
        self.delay = BASE_SCRAPE_DELAY
        self.circuit_state = {'failures': 0, 'last_failure': 0, 'last_success': 0}
        self.metrics = {
            'success': 0, 'failures': 0, 'avg_latency': 0.0,
            'cache_hits': 0, 'cache_misses': 0
        }

        try:
            self.client = MongoClient(config.mongo_uri)
            self.client.admin.command('ping')
            self.db = self.client[config.db_name]
            self.url_cache = self.db['url_cache'] if not config.dry_run else None
            logger.info("Connected to MongoDB at %s, using database: %s", config.mongo_uri, config.db_name)
        except ConnectionFailure as e:
            logger.error("MongoDB connection failed: %s", str(e))
            raise

        if config.debug:
            logger.setLevel(logging.DEBUG)

    def _load_ontology(self) -> set:
        ontology_file = Path(ONTOLOGY_FILE)
        if ontology_file.exists():
            with open(ontology_file) as f:
                return {line.strip().lower() for line in f}
        return {
            'pain', 'fever', 'cough', 'fatigue', 'swelling', 'nausea', 'vomiting',
            # ... default terms
        }

    def _check_circuit(self):
        now = time.time()
        if now - self.circuit_state['last_failure'] > CIRCUIT_RESET_TIME:
            self.circuit_state['failures'] = 0
        if self.circuit_state['failures'] >= CIRCUIT_BREAKER_THRESHOLD:
            raise Exception(f"Circuit breaker tripped: {self.circuit_state['failures']} failures")

    def _record_circuit_failure(self):
        self.circuit_state['failures'] += 1
        self.circuit_state['last_failure'] = time.time()

    def _reset_circuit(self):
        if time.time() - self.circuit_state['last_success'] > CIRCUIT_RESET_TIME:
            self.circuit_state['failures'] = 0

    async def resolve_url(self, base_url: str, condition: str, source: str, session: aiohttp.ClientSession) -> str:
        try:
            self._check_circuit()
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            
            if self.url_cache and not self.config.dry_run:
                if cached := self.url_cache.find_one({'condition': condition, 'source': source}):
                    if await self.verify_url(cached['url'], session):
                        return cached['url']

            if await self._check_robots_allowed(base_url, session) and await self.verify_url(base_url, session):
                self._cache_url(condition, source, base_url)
                return base_url

            search_urls = {
                'mayo_clinic': f"https://www.mayoclinic.org/search/search-results?q={urllib.parse.quote(condition)}",
                'webmd': f"https://www.webmd.com/search/search_results/default.aspx?query={urllib.parse.quote(condition)}",
                'medlineplus': f"https://vsearch.nlm.nih.gov/vivisimo/cgi-bin/query-meta?v%3Aproject=medlineplus&query={urllib.parse.quote(condition)}",
                'cdc': f"https://www.cdc.gov/search/?query={urllib.parse.quote(condition)}"
            }

            if search_url := search_urls.get(source):
                async with session.get(search_url, headers=headers) as response:
                    if response.status == 200:
                        soup = BeautifulSoup(await response.text(), 'html.parser')
                        pattern = re.compile(f'.*{re.escape(condition)}.*', re.I)
                        for link in soup.find_all('a', href=pattern):
                            if resolved_url := self._validate_resolved_url(link['href'], source):
                                return resolved_url

            return self._construct_fallback_url(condition, source)

        except Exception as e:
            logger.error(f"URL resolution error for {condition}: {str(e)}")
            self._record_circuit_failure()
            return self._construct_fallback_url(condition, source)

    def _construct_fallback_url(self, condition: str, source: str) -> str:
        slug = re.sub(r'[^a-z0-9]+', '-', condition.lower()).strip('-')
        mappings = URL_MAPPINGS.get(source, {})
        if source == 'mayo_clinic':
            return f"https://www.mayoclinic.org/diseases-conditions/{slug}/symptoms-causes/syc-{random.randint(20350000, 20359999)}"
        elif source == 'webmd':
            return f"https://www.webmd.com/{slug}"
        elif source == 'medlineplus':
            return f"https://medlineplus.gov/{slug}.html"
        elif source == 'cdc':
            return f"https://www.cdc.gov/{slug}/index.html"
        return ""

    async def scrape_website(self, url: str, condition: str, source: str, session: aiohttp.ClientSession) -> List[str]:
        try:
            cache_key = f"{source}_{condition}_{self._hash_url(url)}"
            if cached := self._get_cached_response(cache_key):
                return cached

            headers = {'User-Agent': random.choice(USER_AGENTS)}
            async with session.get(url, headers=headers) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    await asyncio.sleep(retry_after)
                    return await self.scrape_website(url, condition, source, session)

                if response.status == 200:
                    soup = BeautifulSoup(await response.text(), 'html.parser')
                    symptoms = self._extract_symptoms(soup, source)
                    self._cache_response(cache_key, symptoms)
                    return symptoms
                return []
        except Exception as e:
            logger.error(f"Scraping failed for {url}: {str(e)}")
            return []

    def _extract_symptoms(self, soup: BeautifulSoup, source: str) -> List[str]:
        if source == 'mayo_clinic':
            return self._extract_mayo_clinic_symptoms(soup)
        elif source == 'webmd':
            return self._extract_webmd_symptoms(soup)
        elif source == 'medlineplus':
            return self._extract_medlineplus_symptoms(soup)
        else:
            return self._generic_symptom_extraction(soup)

    def _extract_mayo_clinic_symptoms(self, soup: BeautifulSoup) -> List[str]:
        symptoms = []
        if section := soup.find('div', id='symptoms'):
            symptoms.extend(li.get_text(strip=True) for li in section.find_all('li'))
        return symptoms[:MAX_SYMPTOMS_PER_SOURCE]

    def _extract_webmd_symptoms(self, soup: BeautifulSoup) -> List[str]:
        symptoms = []
        for section in soup.find_all('h2', string=re.compile('symptoms', re.I)):
            if content := section.find_next_sibling('div'):
                symptoms.extend(li.get_text(strip=True) for li in content.find_all('li'))
        return symptoms[:MAX_SYMPTOMS_PER_SOURCE]

    def merge_similar_symptoms(self, symptoms: List[str]) -> List[str]:
        merged = []
        used = set()
        for i, s1 in enumerate(symptoms):
            if i in used:
                continue
            merged.append(s1)
            for j, s2 in enumerate(symptoms[i+1:], i+1):
                if j not in used and fuzz.token_set_ratio(s1, s2) > 80:
                    used.add(j)
        return merged

    async def populate_database(self):
        try:
            collection = self.db[f"{self.config.kb_prefix}diagnosis_relevance"]
            conditions = await self.fetch_disease_list()
            
            async with aiohttp.ClientSession() as session:
                for batch in self._batch_generator(conditions):
                    bulk_ops = []
                    for condition in batch:
                        symptoms = await self._scrape_condition_symptoms(condition, session)
                        if symptoms:
                            bulk_ops.append(UpdateOne(
                                {"condition": condition['name']},
                                {"$set": {"required": symptoms}},
                                upsert=True
                            ))
                    if bulk_ops and not self.config.dry_run:
                        collection.bulk_write(bulk_ops)
                    await asyncio.sleep(self.delay + random.uniform(-0.5, 0.5))
            logger.info("Database population completed")
        except Exception as e:
            logger.error("Population failed: %s", str(e))
        finally:
            self.client.close()

if __name__ == "__main__":
    try:
        logger.info("Starting clinical knowledge population")
        config = Config()
        populator = ClinicalKnowledgePopulator(config)
        asyncio.run(populator.populate_database())
    except Exception as e:
        logger.error("Script failed: %s", str(e))
        sys.exit(1)