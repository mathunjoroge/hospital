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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
import logging
from tqdm import tqdm
from fuzzywuzzy import fuzz
import string
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential
import multiprocessing
import yaml

# Constants
CACHE_TTL = 86400  # 24 hours
REQUEST_TIMEOUT = 45  # Increased to handle slow responses
BASE_SCRAPE_DELAY = 3
MAX_SYMPTOMS_PER_SOURCE = 15
BATCH_SIZE = 10
MIN_SYMPTOMS = 2
MAX_RETRIES = 3
MAX_CACHE_SIZE = 1000
MIN_DELAY = 1
MAX_DELAY = 30
CHECKPOINT_FILE = 'checkpoint.json'

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
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
]

# CDC Conditions
CDC_CONDITIONS = [
    'malaria', 'tuberculosis', 'hiv/aids', 'influenza', 'dengue fever',
    'zika virus', 'cholera', 'alcohol poisoning', 'alcoholic hepatitis'
]

# Load URL mappings from YAML
def load_url_mappings() -> Dict[str, Dict[str, str]]:
    mappings_file = Path('url_mappings.yaml')
    default_mappings = {
        'mayo_clinic': {
            'atrial fibrillation': 'atrial-fibrillation/symptoms-causes/syc-20350624',
            'abdominal aortic aneurysm': 'abdominal-aortic-aneurysm/symptoms-causes/syc-20350688',
            'hyperhidrosis': 'hyperhidrosis/symptoms-causes/syc-20367152',
            'bartholin cyst': 'bartholin-cyst/symptoms-causes/syc-20369976',
            'absence seizure': 'absence-seizure/symptoms-causes/syc-20349684',
            'acanthosis nigricans': 'acanthosis-nigricans/symptoms-causes/syc-20368943',
            'alcoholic hepatitis': 'alcoholic-hepatitis/symptoms-causes/syc-20351388',
            'churg-strauss syndrome': 'eosinophilic-granulomatosis-with-polyangiitis/symptoms-causes/syc-20354558',
            'hay fever': 'hay-fever/symptoms-causes/syc-20373039',
            'alcohol poisoning': 'alcohol-poisoning/symptoms-causes/syc-20354386',
            'allergies': 'allergies/symptoms-causes/syc-20351497',
            'dust mite allergy': 'dust-mites/symptoms-causes/syc-20352178',
            'egg allergy': 'egg-allergy/symptoms-causes/syc-20372115',
            'attention-deficit/hyperactivity disorder (adhd) in children': 'adhd/symptoms-causes/syc-20350889',
            'adult attention-deficit/hyperactivity disorder (adhd)': 'adult-adhd/symptoms-causes/syc-20350878',
            'frozen shoulder': 'frozen-shoulder/symptoms-causes/syc-20372684',
            'adjustment disorders': 'adjustment-disorders/symptoms-causes/syc-20355224',
            'adnexal tumors': 'ovarian-cysts/symptoms-causes/syc-20353405',
            'childhood schizophrenia': 'childhood-schizophrenia/symptoms-causes/syc-20354483',
            'adrenal cancer': 'adrenal-cancer/symptoms-causes/syc-20351026',
            'benign adrenal tumors': 'benign-adrenal-tumors/symptoms-causes/syc-20356190',
            'adrenoleukodystrophy': 'adrenoleukodystrophy/symptoms-causes/syc-20369157',
            'gastroesophageal reflux disease (gerd)': 'gerd/symptoms-causes/syc-20361940',
            'infant reflux': 'infant-acid-reflux/symptoms-causes/syc-20351408',
            'acl injury': 'acl-injury/symptoms-causes/syc-20350738',
            'acne': 'acne/symptoms-causes/syc-20368047',
            'hidradenitis suppurativa': 'hidradenitis-suppurativa/symptoms-causes/syc-20352306',
            'achondroplasia': 'achondroplasia/symptoms-causes/syc-20369188'
        },
        'webmd': {
            'atrial fibrillation': 'heart-disease/atrial-fibrillation',
            'abdominal aortic aneurysm': 'heart-disease/abdominal-aortic-aneurysm',
            'hyperhidrosis': 'skin-problems-and-treatments/hyperhidrosis',
            'bartholin cyst': 'sexual-health/bartholin-cyst',
            'absence seizure': 'epilepsy/absence-seizures',
            'acanthosis nigricans': 'diabetes/acanthosis-nigricans',
            'alcoholic hepatitis': 'hepatitis/alcoholic-hepatitis',
            'churg-strauss syndrome': 'vasculitis/vasculitis-churg-strauss-syndrome',
            'hay fever': 'allergies/hay-fever',
            'alcohol poisoning': 'first-aid/alcohol-poisoning',
            'allergies': 'allergies/default.htm',
            'dust mite allergy': 'allergies/dust-mite-allergy',
            'egg allergy': 'allergies/egg-allergy',
            'attention-deficit/hyperactivity disorder (adhd) in children': 'add-adhd/childhood-adhd',
            'adult attention-deficit/hyperactivity disorder (adhd)': 'add-adhd/adult-adhd',
            'frozen shoulder': 'pain-management/arthritis-frozen-shoulder',
            'adjustment disorders': 'mental-health/adjustment-disorder',
            'adnexal tumors': 'women/guide/ovarian-cysts',
            'childhood schizophrenia': 'schizophrenia/childhood-schizophrenia',
            'adrenal cancer': 'cancer/adrenal-gland-tumor',
            'benign adrenal tumors': 'cancer/adrenal-gland-tumor',
            'adrenoleukodystrophy': 'brain-nervous-system/adrenoleukodystrophy',
            'gastroesophageal reflux disease (gerd)': 'heartburn/gerd-guide/default.htm',
            'infant reflux': 'parenting/guide/infant-reflux',
            'acl injury': 'pain-management/knee-pain/acl-injury',
            'acne': 'skin-problems-and-treatments/acne/default.htm',
            'hidradenitis suppurativa': 'skin-problems-and-treatments/hidradenitis-suppurativa',
            'achondroplasia': 'bones-orthopedics/achondroplasia'
        },
        'medlineplus': {
            'atrial fibrillation': 'atrialfibrillation.html',
            'abdominal aortic aneurysm': 'aorticaneurysm.html',
            'hyperhidrosis': 'sweating.html',
            'bartholin cyst': 'vulvardisorders.html',
            'absence seizure': 'seizures.html',
            'acanthosis nigricans': 'acanthosisnigricans.html',
            'alcoholic hepatitis': 'alcoholicliverdisease.html',
            'churg-strauss syndrome': 'eosinophilicdisorders.html',
            'hay fever': 'allergicrhinitis.html',
            'alcohol poisoning': 'alcohol.html',
            'allergies': 'allergy.html',
            'dust mite allergy': 'dustmites.html',
            'egg allergy': 'foodallergy.html',
            'attention-deficit/hyperactivity disorder (adhd) in children': 'attentiondeficithyperactivitydisorder.html',
            'adult attention-deficit/hyperactivity disorder (adhd)': 'attentiondeficithyperactivitydisorder.html',
            'frozen shoulder': 'shoulderinjuriesanddisorders.html',
            'adjustment disorders': 'mentaldisorders.html',
            'adnexal tumors': 'ovariancysts.html',
            'childhood schizophrenia': 'schizophrenia.html',
            'adrenal cancer': 'adrenalglandcancer.html',
            'benign adrenal tumors': 'adrenalglanddisorders.html',
            'adrenoleukodystrophy': 'adrenoleukodystrophy.html',
            'gastroesophageal reflux disease (gerd)': 'gerd.html',
            'infant reflux': 'infantandnewborncare.html',
            'acl injury': 'kneeinjuriesanddisorders.html',
            'acne': 'acne.html',
            'hidradenitis suppurativa': 'hidradenitissuppurativa.html',
            'achondroplasia': 'genetics/condition/achondroplasia'
        }
    }
    
    if mappings_file.exists():
        with mappings_file.open('r') as f:
            mappings = yaml.safe_load(f) or default_mappings
    else:
        mappings = default_mappings
        with mappings_file.open('w') as f:
            yaml.safe_dump(mappings, f)
    return mappings

URL_MAPPINGS = load_url_mappings()

class Config:
    def __init__(self):
        load_dotenv()
        self.mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
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
        logger.debug("Initializing ClinicalKnowledgePopulator")
        self.config = config
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.failed_fetches = []
        self.executor = ThreadPoolExecutor(max_workers=max(4, multiprocessing.cpu_count()))
        self.medical_terms = self._load_medical_terms()
        self.delay = BASE_SCRAPE_DELAY
        
        if not config.mongo_uri:
            logger.error("MONGO_URI is not set")
            raise ValueError("MONGO_URI environment variable is not set")
        
        try:
            logger.debug("Connecting to MongoDB at %s", config.mongo_uri)
            self.client = MongoClient(config.mongo_uri)
            self.client.admin.command('ping')
            logger.info("Connected to MongoDB at %s", config.mongo_uri)
            self.db = self.client[config.db_name]
            if config.db_name not in self.client.list_database_names():
                logger.error("Database %s does not exist", config.db_name)
                raise ValueError(f"Database {config.db_name} does not exist")
            logger.info("Using database: %s", config.db_name)
            
            self.url_cache = self.db['url_cache'] if not config.dry_run else None
            logger.debug("Initialized url_cache: %s", self.url_cache)
        except ConnectionFailure as e:
            logger.error("MongoDB connection failed: %s", str(e))
            self.db = None
            self.url_cache = None
            raise

        if config.debug:
            logger.setLevel(logging.DEBUG)

    def _load_medical_terms(self) -> set:
        return {
            'pain', 'fever', 'cough', 'fatigue', 'swelling', 'nausea', 'vomiting',
            'diarrhea', 'constipation', 'bleeding', 'rash', 'itch', 'dizziness',
            'weakness', 'numbness', 'tingling', 'shortness of breath', 'chest',
            'headache', 'confusion', 'sleep', 'appetite', 'weight', 'thirst',
            'urination', 'vision', 'hearing', 'speech', 'movement', 'tremor',
            'palpitations', 'seizure', 'sweating', 'skin darkening', 'staring spells',
            'swelling near vagina', 'back pain', 'abdominal pain', 'discoloration',
            'bruising', 'tenderness', 'redness', 'irregular heartbeat', 'pulsating sensation',
            'excessive sweating', 'darkened skin', 'staring blankly', 'swelling near vaginal opening',
            'velvety skin', 'sudden stop in movement', 'brief loss of awareness',
            'jaundice', 'nasal congestion', 'runny nose', 'sneezing', 'itchy eyes', 'wheezing',
            'sore throat', 'joint pain', 'muscle pain', 'chills', 'paleness', 'hair loss',
            'night sweats', 'difficulty swallowing', 'hoarseness', 'persistent cough',
            'inattention', 'hyperactivity', 'impulsivity', 'stiffness', 'sadness', 'anxiety',
            'depression', 'behavioral changes', 'hallucinations', 'delusions', 'weight gain',
            'weight loss', 'high blood pressure', 'muscle weakness', 'coordination problems',
            'heartburn', 'regurgitation', 'spitting up', 'irritability', 'popping sensation',
            'instability', 'blackheads', 'whiteheads', 'pimples', 'nodules', 'cysts',
            'painful lumps', 'abscesses', 'short stature', 'large head', 'bowed legs'
        }

    async def fetch_disease_list(self) -> List[Dict[str, str]]:
        diseases = []
        
        mayo_diseases = await self._fetch_mayo_disease_list()
        diseases.extend([{'name': d['name'], 'mayo_key': d['key']} for d in mayo_diseases])
        
        webmd_diseases = await self._fetch_webmd_disease_list()
        for wd in webmd_diseases:
            match = self._find_matching_disease(wd['name'], diseases)
            if match:
                match['webmd_key'] = wd['key']
            else:
                diseases.append({'name': wd['name'], 'webmd_key': wd['key']})
        
        medline_diseases = await self._fetch_medlineplus_disease_list()
        for md in medline_diseases:
            match = self._find_matching_disease(md['name'], diseases)
            if match:
                match['medlineplus_key'] = md['key']
            else:
                diseases.append({'name': md['name'], 'medlineplus_key': md['key']})
        
        normalized_diseases = []
        used_names = set()
        for d in diseases:
            name = self._normalize_condition_name(d['name'])
            if name not in used_names:
                normalized_diseases.append({
                    'name': name,
                    'mayo_key': d.get('mayo_key') or URL_MAPPINGS['mayo_clinic'].get(name) or name.replace(' ', '-').lower(),
                    'webmd_key': d.get('webmd_key') or URL_MAPPINGS['webmd'].get(name) or f"{name.replace(' ', '-').lower()}",
                    'medlineplus_key': d.get('medlineplus_key') or URL_MAPPINGS['medlineplus'].get(name) or name.replace(' ', '').lower()
                })
                used_names.add(name)
        
        logger.info("Compiled %d unique diseases", len(normalized_diseases))
        return normalized_diseases

    async def _fetch_mayo_disease_list(self) -> List[Dict[str, str]]:
        url = "https://www.mayoclinic.org/diseases-conditions/index"
        diseases = []
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'User-Agent': random.choice(USER_AGENTS)}
                async with session.get(url, timeout=REQUEST_TIMEOUT, headers=headers) as response:
                    if response.status != 200:
                        logger.warning("Failed to fetch Mayo Clinic disease list: Status %d", response.status)
                        return []
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    for a in soup.find_all('a', href=re.compile(r'/diseases-conditions/[^/]+/symptoms-causes')):
                        name = a.get_text(strip=True).lower()
                        key = '/'.join(a['href'].split('/')[-3:-1])
                        diseases.append({'name': name, 'key': key})
        except aiohttp.ClientError as e:
            logger.error("Error fetching Mayo Clinic disease list: %s", str(e))
        return diseases

    async def _fetch_webmd_disease_list(self) -> List[Dict[str, str]]:
        url = "https://www.webmd.com/a-to-z-guides/health-topics"
        diseases = []
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'User-Agent': random.choice(USER_AGENTS)}
                async with session.get(url, timeout=REQUEST_TIMEOUT, headers=headers) as response:
                    if response.status != 200:
                        logger.warning("Failed to fetch WebMD disease list: Status %d", response.status)
                        return []
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    for a in soup.find_all('a', href=re.compile(r'webmd.com/')):
                        name = a.get_text(strip=True).lower()
                        key = a['href'].replace('https://www.webmd.com/', '').rstrip('/')
                        if key and name:
                            diseases.append({'name': name, 'key': key})
        except aiohttp.ClientError as e:
            logger.error("Error fetching WebMD disease list: %s", str(e))
        return diseases

    async def _fetch_medlineplus_disease_list(self) -> List[Dict[str, str]]:
        url = "https://medlineplus.gov/healthtopics.html"
        diseases = []
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'User-Agent': random.choice(USER_AGENTS)}
                async with session.get(url, timeout=REQUEST_TIMEOUT, headers=headers) as response:
                    if response.status != 200:
                        logger.warning("Failed to fetch MedlinePlus disease list: Status %d", response.status)
                        return []
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    for a in soup.find_all('a', href=re.compile(r'medlineplus.gov/[^/]+\.html')):
                        name = a.get_text(strip=True).lower()
                        key = a['href'].replace('https://medlineplus.gov/', '').replace('.html', '')
                        if key and name:
                            diseases.append({'name': name, 'key': key})
        except aiohttp.ClientError as e:
            logger.error("Error fetching MedlinePlus disease list: %s", str(e))
        return diseases

    def _normalize_condition_name(self, name: str) -> str:
        name = name.lower().strip()
        name = re.sub(r'\s+', ' ', name)
        name = re.sub(r'[^\w\s-]', '', name)
        replacements = {
            'bartholins cyst': 'bartholin cyst',
            'type 2 diabetes': 'diabetes mellitus type 2',
            'hiv aids': 'hiv/aids',
            'copd': 'chronic obstructive pulmonary disease',
            'eosinophilic granulomatosis with polyangiitis': 'churg-strauss syndrome',
            'attention-deficithyperactivity disorder adhd in children': 'attention-deficit/hyperactivity disorder (adhd) in children',
            'adult attention-deficithyperactivity disorder adhd': 'adult attention-deficit/hyperactivity disorder (adhd)',
            'adnexal tumors and masses': 'adnexal tumors',
            'gastroesophageal reflux disease gerd': 'gastroesophageal reflux disease (gerd)'
        }
        for old, new in replacements.items():
            if old in name:
                return new
        return name

    def _find_matching_disease(self, name: str, diseases: List[Dict[str, str]], threshold: int = 90) -> Optional[Dict[str, str]]:
        name = self._normalize_condition_name(name)
        for d in diseases:
            if fuzz.ratio(name, self._normalize_condition_name(d['name'])) > threshold:
                return d
        return None

    async def resolve_url(self, base_url: str, condition: str, source: str, session: aiohttp.ClientSession) -> str:
        logger.debug("Resolving URL for %s (%s): %s", condition, source, base_url)
        if self.url_cache is not None and not self.config.dry_run:
            cached_url = self.url_cache.find_one({'condition': condition, 'source': source})
            if cached_url and await self.verify_url(cached_url['url'], session):
                logger.debug("Using cached URL for %s (%s): %s", condition, source, cached_url['url'])
                return cached_url['url']
        
        if await self.verify_url(base_url, session):
            self._cache_url(condition, source, base_url)
            return base_url
        
        search_urls = {
            'mayo_clinic': f"https://www.mayoclinic.org/search/search-results?q={urllib.parse.quote(condition)}",
            'webmd': f"https://www.webmd.com/search/search_results/default.aspx?query={urllib.parse.quote(condition)}",
            'medlineplus': f"https://vsearch.nlm.nih.gov/vivisimo/cgi-bin/query-meta?v%3Aproject=medlineplus&query={urllib.parse.quote(condition)}",
            'cdc': f"https://www.cdc.gov/search/?query={urllib.parse.quote(condition)}"
        }
        
        search_url = search_urls.get(source)
        if not search_url:
            logger.debug("No search URL for %s, using fallback", source)
            return self._construct_fallback_url(condition, source)
        
        try:
            async with session.get(search_url, timeout=REQUEST_TIMEOUT) as response:
                if response.status != 200:
                    logger.warning("Search failed for %s (%s): Status %d", condition, source, response.status)
                    return self._construct_fallback_url(condition, source)
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                pattern = f'.*{condition.replace(" ", "[ -]").lower()}.*'
                valid_domains = {
                    'mayo_clinic': 'mayoclinic.org',
                    'webmd': 'webmd.com',
                    'medlineplus': 'medlineplus.gov',
                    'cdc': 'cdc.gov'
                }
                
                valid_domain = valid_domains.get(source)
                for link in soup.find_all('a', href=re.compile(pattern, re.I)):
                    href = link.get('href', '')
                    if valid_domain in href and 'search' not in href.lower():
                        resolved_url = href if href.startswith('http') else f"https://{valid_domain}{href}"
                        if await self.verify_url(resolved_url, session):
                            logger.debug("Resolved %s URL for %s: %s", source, condition, resolved_url)
                            self._cache_url(condition, source, resolved_url)
                            return resolved_url
                
                logger.warning("No valid %s URL found in search for %s", source, condition)
                return self._construct_fallback_url(condition, source)
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning("Failed to resolve URL for %s (%s) via search: %s", condition, source, str(e))
            self._record_failure(condition, source, f"URL resolution failed: {str(e)}")
            return self._construct_fallback_url(condition, source)

    async def verify_url(self, url: str, session: aiohttp.ClientSession) -> bool:
        try:
            async with session.head(url, timeout=5, allow_redirects=True) as response:
                return response.status == 200
        except:
            return False

    def _cache_url(self, condition: str, source: str, url: str) -> None:
        if self.url_cache is not None and not self.config.dry_run:
            try:
                self.url_cache.update_one(
                    {'condition': condition, 'source': source},
                    {'$set': {'url': url, 'timestamp': datetime.now()}},
                    upsert=True
                )
                logger.debug("Cached URL for %s (%s): %s", condition, source, url)
            except PyMongoError as e:
                logger.warning("Failed to cache URL for %s (%s): %s", condition, source, str(e))

    def _construct_fallback_url(self, condition: str, source: str) -> str:
        condition = self._normalize_condition_name(condition)
        mappings = URL_MAPPINGS.get(source, {})
        
        if source == 'mayo_clinic':
            key = mappings.get(condition, condition.replace(' ', '-').lower())
            return f"https://www.mayoclinic.org/diseases-conditions/{key}/symptoms-causes/syc-{random.randint(20350000, 20359999)}"
        elif source == 'webmd':
            key = mappings.get(condition, condition.replace(' ', '-').lower())
            return f"https://www.webmd.com/{key}"
        elif source == 'medlineplus':
            key = mappings.get(condition, f"{condition.replace(' ', '').lower()}")
            return f"https://medlineplus.gov/{key}.html"
        elif source == 'cdc':
            key = condition.replace(' ', '-').lower()
            return f"https://www.cdc.gov/{key}/index.html"
        return ""

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def scrape_website(self, url: str, condition: str, source: str, session: aiohttp.ClientSession) -> List[str]:
        if not url:
            return []
            
        cache_key = f"{source}_{condition}_{self._hash_url(url)}"
        cached, last_modified = self._get_cached_response(cache_key)
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        
        if last_modified:
            headers['If-Modified-Since'] = last_modified
        
        try:
            logger.info("Scraping %s URL for %s: %s", source, condition, url)
            async with session.head(url, timeout=REQUEST_TIMEOUT, headers=headers, allow_redirects=True) as head_response:
                if head_response.status == 304 and cached:
                    logger.info("Using cached symptoms for %s from %s (not modified)", condition, source)
                    return cached
                
                if head_response.status != 200:
                    logger.warning("Invalid URL %s for %s from %s: Status %d", url, condition, source, head_response.status)
                    self._record_failure(condition, source, f"HTTP {head_response.status}")
                    return []
                
                last_modified = head_response.headers.get('Last-Modified')
                
                async with session.get(url, timeout=REQUEST_TIMEOUT, headers=headers) as response:
                    if response.status != 200:
                        logger.warning("Failed to fetch %s for %s: Status %d", url, condition, response.status)
                        self._record_failure(condition, source, f"HTTP {response.status}")
                        return []
                    
                    html = await response.text()
                    loop = asyncio.get_event_loop()
                    symptoms = await loop.run_in_executor(
                        self.executor,
                        self._parse_html,
                        html,
                        source
                    )
                    
                    if symptoms:
                        logger.info("Scraped %d symptoms for %s from %s", len(symptoms), condition, source)
                    else:
                        logger.warning("No symptoms extracted for %s from %s", condition, source)
                        self._record_failure(condition, source, "No symptoms extracted")
                    
                    self._cache_response(cache_key, symptoms, last_modified)
                    return symptoms
                    
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error("Network error scraping %s for %s at %s: %s", source, condition, url, str(e))
            self._record_failure(condition, source, str(e))
            return []
        except Exception as e:
            logger.error("Unexpected error scraping %s for %s: %s", source, condition, str(e))
            return []

    def _hash_url(self, url: str) -> str:
        return hashlib.sha256(url.encode('utf-8')).hexdigest()[:32]

    def _parse_html(self, html: str, source: str) -> List[str]:
        soup = BeautifulSoup(html, 'html.parser')
        try:
            return self._extract_symptoms(soup, source)
        except Exception as e:
            logger.error("Error parsing HTML for %s: %s", source, str(e))
            return []

    def _extract_symptoms(self, soup: BeautifulSoup, source: str) -> List[str]:
        symptoms = []
        methods = [
            self._extract_from_ul_lists,
            self._extract_from_symptom_sections,
            self._extract_from_paragraphs,
            self._extract_from_divs
        ]
        
        for method in methods:
            try:
                extracted = method(soup, source)
                if extracted:
                    symptoms.extend(extracted)
                    logger.debug("Extracted %d symptoms using %s for %s", len(extracted), method.__name__, source)
            except Exception as e:
                logger.warning("Extraction method %s failed for %s: %s", method.__name__, source, str(e))
        
        if not symptoms:
            logger.warning("All extraction methods failed for %s", source)
        return symptoms[:MAX_SYMPTOMS_PER_SOURCE]

    def _extract_from_ul_lists(self, soup: BeautifulSoup, source: str) -> List[str]:
        symptoms = []
        for ul in soup.find_all('ul'):
            for li in ul.find_all('li'):
                text = li.get_text(strip=True).lower()
                if any(term in text for term in self.medical_terms) or 'symptom' in text:
                    symptoms.append(text.split('.')[0].strip())
        return symptoms

    def _extract_from_symptom_sections(self, soup: BeautifulSoup, source: str) -> List[str]:
        symptoms = []
        symptom_keywords = ['symptom', 'sign and symptom', 'clinical presentation', 'manifestation', 'signs', 'symptoms include']
        
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b'])
        for header in headers:
            header_text = header.get_text(strip=True).lower()
            if any(keyword in header_text for keyword in symptom_keywords):
                next_node = header.next_sibling
                while next_node:
                    if next_node.name == 'ul':
                        symptoms.extend([li.get_text(strip=True).lower() for li in next_node.find_all('li')])
                    elif next_node.name in ['p', 'div']:
                        text = next_node.get_text(strip=True).lower()
                        if any(term in text for term in self.medical_terms) or 'symptom' in text:
                            symptoms.append(text.split('.')[0].strip())
                    elif next_node.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        break
                    next_node = next_node.next_sibling
        return symptoms

    def _extract_from_paragraphs(self, soup: BeautifulSoup, source: str) -> List[str]:
        symptoms = []
        content_containers = ['article', 'main', 'div.content', 'div.main-content', 'section']
        
        for container in content_containers:
            content = soup.find(container.split('.')[0], class_=container.split('.')[1] if '.' in container else None)
            if content:
                for p in content.find_all(['p', 'div']):
                    text = p.get_text(strip=True).lower()
                    if any(term in text for term in self.medical_terms) or 'symptom' in text:
                        sentences = re.split(r'[.,;]', text)
                        symptoms.extend([s.strip() for s in sentences if s.strip() and len(s.strip()) > 5])
        return symptoms

    def _extract_from_divs(self, soup: BeautifulSoup, source: str) -> List[str]:
        symptoms = []
        for div in soup.find_all('div', class_=re.compile(r'symptom|content|health-info|article|section', re.I)):
            text = div.get_text(strip=True).lower()
            if any(term in text for term in self.medical_terms) or 'symptom' in text:
                sentences = re.split(r'[.,;]', text)
                symptoms.extend([s.strip() for s in sentences if s.strip() and len(s.strip()) > 5])
        return symptoms

    def _process_symptoms(self, symptoms: List[str]) -> List[str]:
        filtered = [
            s for s in symptoms
            if len(s) > 3 and
            all(phrase not in s.lower() for phrase in [
                'see a doctor', 'call', 'emergency', 'contact', 'mayo clinic',
                'cdc', 'webmd', 'medlineplus', 'treatment', 'diagnosis',
                'learn more', 'overview', 'prevention', 'visit', 'website',
                'test', 'history', 'family', 'risk', 'cause', 'condition', 'manage',
                'lifestyle', 'medicine', 'surgery', 'procedure', 'device', 'transplant'
            ])
        ]
        filtered = list(dict.fromkeys(filtered))
        return self.merge_similar_symptoms(filtered)

    def merge_similar_symptoms(self, symptoms: List[str], threshold: int = 85) -> List[str]:
        merged = []
        used = set()
        for i, s1 in enumerate(symptoms):
            if i in used:
                continue
            merged.append(s1)
            used.add(i)
            for j, s2 in enumerate(symptoms[i+1:], i+1):
                if j not in used and fuzz.ratio(s1.lower(), s2.lower()) > threshold:
                    used.add(j)
        return merged

    def _get_cached_response(self, key: str) -> Tuple[Optional[List[str]], Optional[str]]:
        sanitized_key = self._sanitize_filename(key)
        cache_file = self.config.cache_dir / f"{sanitized_key}.json.gz"
        try:
            if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < CACHE_TTL:
                with gzip.open(cache_file, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                self.cache_hits += 1
                if isinstance(data, list):
                    return data, None
                return data.get('symptoms'), data.get('last_modified')
        except Exception as e:
            logger.warning("Error reading cache file %s: %s", cache_file, str(e))
        self.cache_misses += 1
        return None, None

    def _cache_response(self, key: str, data: List[str], last_modified: Optional[str]) -> None:
        sanitized_key = self._sanitize_filename(key)
        cache_file = self.config.cache_dir / f"{sanitized_key}.json.gz"
        try:
            self._evict_cache()
            with gzip.open(cache_file, 'wt', encoding='utf-8') as f:
                json.dump({'symptoms': data, 'last_modified': last_modified}, f)
        except Exception as e:
            logger.warning("Failed to cache response to %s: %s", cache_file, str(e))

    def _evict_cache(self) -> None:
        cache_files = list(self.config.cache_dir.glob('*.json.gz'))
        if len(cache_files) > MAX_CACHE_SIZE:
            cache_files.sort(key=lambda x: x.stat().st_atime)
            for old_file in cache_files[:len(cache_files) - MAX_CACHE_SIZE]:
                try:
                    old_file.unlink()
                    logger.debug("Evicted old cache file: %s", old_file)
                except Exception as e:
                    logger.warning("Failed to evict cache file %s: %s", old_file, str(e))

    def _sanitize_filename(self, filename: str) -> str:
        if len(filename) > 100:
            return hashlib.sha256(filename.encode('utf-8')).hexdigest()[:32]
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        return ''.join(c if c in valid_chars else '_' for c in filename)

    def _record_failure(self, condition: str, source: str, error: str) -> None:
        self.failed_fetches.append({
            "condition": condition,
            "source": source,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.failed_fetches) % 15 == 0:  # Further increased threshold
            self.delay = min(self.delay * 1.2, MAX_DELAY)
            logger.info("Increased delay to %s seconds due to failures", self.delay)

    def _load_checkpoint(self) -> int:
        checkpoint_path = self.config.backup_dir / CHECKPOINT_FILE
        if checkpoint_path.exists():
            try:
                with checkpoint_path.open('r') as f:
                    data = json.load(f)
                return data.get('last_batch', 0)
            except Exception as e:
                logger.warning("Failed to load checkpoint: %s", str(e))
        return 0

    def _save_checkpoint(self, batch_index: int) -> None:
        checkpoint_path = self.config.backup_dir / CHECKPOINT_FILE
        try:
            with checkpoint_path.open('w') as f:
                json.dump({'last_batch': batch_index}, f)
            logger.debug("Saved checkpoint for batch %d", batch_index)
        except Exception as e:
            logger.warning("Failed to save checkpoint: %s", str(e))

    async def populate_database(self):
        try:
            if self.db is None:
                raise ValueError("Database connection not initialized")
            logger.debug("Database connection validated: %s", self.db.name)
            db = self.db
            await self._update_diagnosis_relevance(db)
            logger.info("Database population completed")
        except Exception as e:
            logger.error("Database population failed: %s", str(e))
            raise
        finally:
            self.client.close()

    async def _update_diagnosis_relevance(self, db):
        collection_name = f"{self.config.kb_prefix}diagnosis_relevance"
        logger.info("Accessing collection: %s", collection_name)
        collection = db[collection_name]
        
        try:
            if not self.config.dry_run:
                self._backup_collection(db, collection_name)
            
            conditions = await self.fetch_disease_list()
            diagnosis_data = {}
            
            start_batch = self._load_checkpoint()
            logger.info("Resuming from batch %d", start_batch)
            
            async with aiohttp.ClientSession() as session:
                for i in tqdm(range(start_batch * BATCH_SIZE, len(conditions), BATCH_SIZE), desc="Processing batches"):
                    batch = conditions[i:i + BATCH_SIZE]
                    logger.debug("Processing batch %d with %d conditions", i // BATCH_SIZE + 1, len(batch))
                    for condition in batch:
                        try:
                            symptoms = await self._scrape_condition_symptoms(condition, session)
                            if symptoms and len(symptoms) >= MIN_SYMPTOMS:
                                diagnosis_data[condition['name']] = symptoms
                                if not self.config.dry_run:
                                    collection.update_one(
                                        {"condition": condition['name']},
                                        {"$set": {"required": symptoms}},
                                        upsert=True
                                    )
                            else:
                                logger.warning("Insufficient symptoms (%d) for %s", len(symptoms), condition['name'])
                                self._record_failure(condition['name'], "all_sources", f"Too few symptoms: {len(symptoms)}")
                        except Exception as e:
                            logger.error("Error processing %s: %s", condition['name'], str(e))
                            self._record_failure(condition['name'], "all_sources", f"Processing error: {str(e)}")
                        
                        await asyncio.sleep(self.delay + random.uniform(0, 1))
                    
                    self._save_checkpoint(i // BATCH_SIZE + 1)
            
            if diagnosis_data:
                logger.info("Processed %d documents for %s", len(diagnosis_data), collection_name)
                if not self.config.dry_run:
                    collection.create_index("condition", unique=True)
                    collection.create_index([("required", 1)])
                    logger.info("Created indexes on condition and required fields")
                
            if self.failed_fetches:
                self._save_failed_fetches()
                
        except PyMongoError as e:
            logger.error("Failed to update %s: %s", collection_name, str(e))
            raise

    async def _scrape_condition_symptoms(self, condition: Dict[str, str], session: aiohttp.ClientSession) -> List[str]:
        symptoms = []
        condition_name = condition['name']
        
        tasks = [
            self.scrape_website(
                await self.resolve_url(
                    f"https://www.mayoclinic.org/diseases-conditions/{condition.get('mayo_key')}",
                    condition_name, "mayo_clinic", session
                ),
                condition_name, "mayo_clinic", session
            ),
            self.scrape_website(
                await self.resolve_url(
                    f"https://www.webmd.com/{condition.get('webmd_key')}",
                    condition_name, "webmd", session
                ),
                condition_name, "webmd", session
            ),
            self.scrape_website(
                await self.resolve_url(
                    f"https://medlineplus.gov/{condition.get('medlineplus_key')}",
                    condition_name, "medlineplus", session
                ),
                condition_name, "medlineplus", session
            )
        ]
        
        if condition_name.lower() in [c.lower() for c in CDC_CONDITIONS]:
            cdc_keys = {
                'malaria': 'malaria/about/index.html',
                'tuberculosis': 'tb/index.html',
                'hiv/aids': 'hiv/index.html',
                'influenza': 'flu/index.html',
                'dengue fever': 'dengue/index.html',
                'zika virus': 'zika/index.html',
                'cholera': 'cholera/index.html',
                'alcohol poisoning': 'alcohol/features/alcohol-poisoning.html',
                'alcoholic hepatitis': 'alcohol/features/alcoholic-liver-disease.html'
            }
            key = cdc_keys.get(condition_name.lower(), f"{condition_name.lower().replace(' ', '-')}/index.html")
            tasks.append(
                self.scrape_website(
                    await self.resolve_url(
                        f"https://www.cdc.gov/{key}",
                        condition_name, "cdc", session
                    ),
                    condition_name, "cdc", session
                )
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                symptoms.extend(result)
        
        processed_symptoms = self._process_symptoms(symptoms)
        if len(processed_symptoms) < MIN_SYMPTOMS:
            logger.warning("Insufficient symptoms (%d) for %s after processing", len(processed_symptoms), condition_name)
            self._record_failure(condition_name, "all_sources", f"Too few symptoms: {len(processed_symptoms)}")
            return await self._cleveland_fallback_search(condition_name, session)
        
        return processed_symptoms

    async def _cleveland_fallback_search(self, condition: str, session: aiohttp.ClientSession) -> List[str]:
        logger.warning("Attempting Cleveland Clinic fallback for %s", condition)
        search_url = f"https://my.clevelandclinic.org/health/diseases?search={urllib.parse.quote(condition)}"
        
        try:
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            async with session.get(search_url, headers=headers, timeout=REQUEST_TIMEOUT) as response:
                if response.status != 200:
                    logger.warning("Cleveland Clinic search failed for %s: Status %d", condition, response.status)
                    return []
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                symptoms = []
                for link in soup.find_all('a', href=re.compile(r'/health/diseases/\d+')):
                    disease_url = f"https://my.clevelandclinic.org{link['href']}"
                    async with session.get(disease_url, headers=headers, timeout=REQUEST_TIMEOUT) as disease_response:
                        if disease_response.status != 200:
                            continue
                        disease_html = await disease_response.text()
                        disease_soup = BeautifulSoup(disease_html, 'html.parser')
                        
                        for method in [self._extract_from_ul_lists, self._extract_from_symptom_sections, self._extract_from_paragraphs, self._extract_from_divs]:
                            try:
                                extracted = method(disease_soup, 'cleveland_clinic')
                                if extracted:
                                    symptoms.extend(extracted)
                                    logger.debug("Extracted %d symptoms using %s for Cleveland Clinic", len(extracted), method.__name__)
                            except Exception as e:
                                logger.warning("Cleveland Clinic extraction method %s failed: %s", method.__name__, str(e))
                
                processed = self._process_symptoms(symptoms)
                logger.info("Cleveland Clinic fallback extracted %d symptoms for %s", len(processed), condition)
                return processed[:MAX_SYMPTOMS_PER_SOURCE]
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error("Cleveland Clinic fallback failed for %s: %s", condition, str(e))
            return []

    def _backup_collection(self, db, collection_name: str) -> bool:
        try:
            backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.config.backup_dir / f"db_backup_{backup_time}"
            backup_path.mkdir(exist_ok=True)
            
            with open(backup_path / f"{collection_name}.json", 'w') as f:
                cursor = db[collection_name].find({})
                f.write('[\n')
                first = True
                for doc in cursor:
                    if not first:
                        f.write(',\n')
                    json.dump(dict(doc, _id=str(doc['_id'])), f)
                    first = False
                f.write('\n]')
            
            logger.info("Backed up %s to %s", collection_name, backup_path)
            return True
        except Exception as e:
            logger.error("Collection backup failed for %s: %s", collection_name, str(e))
            return False

    def _save_failed_fetches(self):
        try:
            with (self.config.backup_dir / 'failed_fetches.json').open('w') as f:
                json.dump(self.failed_fetches, f, indent=2)
            logger.info("Saved %d failed fetches", len(self.failed_fetches))
        except Exception as e:
            logger.error("Failed to save failed fetches: %s", str(e))

    def __del__(self):
        if hasattr(self, 'client'):
            self.client.close()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

if __name__ == "__main__":
    try:
        logger.info("Starting clinical knowledge population")
        config = Config()
        populator = ClinicalKnowledgePopulator(config)
        asyncio.run(populator.populate_database())
    except Exception as e:
        logger.error("Script failed: %s", str(e), exc_info=True)
        sys.exit(1)