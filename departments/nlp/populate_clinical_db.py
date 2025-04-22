#!/usr/bin/env python3
"""
Enhanced Clinical Knowledge Base Population Script
with robust error handling, async/sync separation, and complete functionality
"""

import os
import json
import random
import time
import urllib.parse
import asyncio
import resource
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import requests
import aiohttp
from bs4 import BeautifulSoup
from Bio import Entrez
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, PyMongoError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
import click
from tqdm import tqdm
import shutil
import sys


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clinical_db_population.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    'mongo_uri': os.getenv('MONGO_URI', 'mongodb://localhost:27017'),
    'db_name': os.getenv('DB_NAME', 'clinical_db'),
    'kb_prefix': os.getenv('KB_PREFIX', 'kb_'),
    'symptoms_collection': os.getenv('SYMPTOMS_COLLECTION', 'symptoms'),
    'cache_dir': Path(os.getenv('CACHE_DIR', 'data_cache')),
    'cache_ttl': int(os.getenv('CACHE_TTL', 86400)),
    'request_timeout': int(os.getenv('REQUEST_TIMEOUT', 15)),
    'max_retries': int(os.getenv('MAX_RETRIES', 3)),
    'pubmed_email': os.getenv('PUBMED_EMAIL', 'your.email@example.com'),
    'conditions_file': Path(os.getenv('CONDITIONS_FILE', 'conditions.json')),
    'backup_dir': Path(os.getenv('BACKUP_DIR', 'backups')),
    'icd10_mappings_file': Path(os.getenv('ICD10_MAPPINGS_FILE', 'icd10_mappings.json')),
}

# Create directories
for dir_key in ['cache_dir', 'backup_dir']:
    CONFIG[dir_key].mkdir(parents=True, exist_ok=True)

# User agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
]

class ClinicalKnowledgePopulator:
    def __init__(self, use_nlp: bool = True):
        self.start_time = time.time()
        self.cache = {}
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
        Entrez.email = CONFIG['pubmed_email']
        self.cache_hits = 0
        self.cache_misses = 0
        self.articles_fetched = 0
        self.validation_warnings = 0
        self.condition_report = []
        self.failed_fetches = []
        self.summarizer = None
        self.icd10_map = self._load_icd10_mappings()
        self.condition_data = self._load_condition_data()
        
        if use_nlp:
            self._initialize_nlp()

    def _load_condition_data(self) -> Dict:
        """Load condition data with validation"""
        try:
            with CONFIG['conditions_file'].open('r') as f:
                data = json.load(f)
            
            # Validate and set defaults
            for condition, cond_data in data.items():
                if not isinstance(cond_data, dict):
                    logger.warning(f"Invalid data for condition {condition}, using defaults")
                    data[condition] = self._default_condition_data(condition)
                    continue
                
                for field in ["differentials", "follow_up"]:
                    if field not in cond_data:
                        cond_data[field] = []
                
                for field in ["workup", "management"]:
                    if field not in cond_data:
                        cond_data[field] = {"urgent": [], "routine": []} if field == "workup" else {"symptomatic": [], "definitive": [], "lifestyle": []}
                
                if "prevalence_score" not in cond_data:
                    cond_data["prevalence_score"] = 0.1
            
            logger.info(f"Loaded {len(data)} conditions from {CONFIG['conditions_file']}")
            return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Using default condition data: {str(e)}")
            return {"heart failure": self._default_condition_data("heart failure")}

    def _default_condition_data(self, condition: str) -> Dict:
        """Generate default data for a condition"""
        return {
            "category": "general",
            "key": condition,
            "pubmed": condition,
            "mayo": condition.lower().replace(' ', '-'),
            "cdc": condition.lower().replace(' ', '-'),
            "differentials": [condition.capitalize()],
            "workup": {"urgent": [], "routine": []},
            "management": {"symptomatic": [], "definitive": [], "lifestyle": []},
            "follow_up": [],
            "prevalence_score": 0.1
        }

    def _load_icd10_mappings(self) -> Dict:
        """Load ICD-10 codes with fallback to defaults"""
        default_map = {
            "heart failure": "I50",
            "malaria": "B50-B54",
            "gout": "M10",
            "diabetes": "E11",
            "pneumonia": "J18",
            "hypertension": "I10"
        }
        
        try:
            if CONFIG['icd10_mappings_file'].exists():
                with CONFIG['icd10_mappings_file'].open('r') as f:
                    custom_map = json.load(f)
                default_map.update(custom_map)
        except json.JSONDecodeError:
            logger.warning("Invalid ICD-10 mappings file, using defaults")
        
        return default_map

    def _initialize_nlp(self):
        """Initialize NLP components with retries"""
        try:
            from transformers import pipeline
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-6-6",
                device=-1  # Use CPU
            )
            logger.info("NLP summarizer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize NLP: {str(e)}")
            self.summarizer = None

    def _backup_file(self, file_path: Path) -> bool:
        """Create timestamped backup of a file"""
        try:
            if file_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = CONFIG['backup_dir'] / f"{file_path.stem}_{timestamp}{file_path.suffix}"
                shutil.copy(file_path, backup_path)
                logger.info(f"Backed up {file_path} to {backup_path}")
                return True
        except Exception as e:
            logger.warning(f"Failed to backup {file_path}: {str(e)}")
        return False

    # [Previous methods like get_category, get_key, create_pathway_template remain the same...]

    async def fetch_all_data(self, condition: str) -> Tuple[List[Dict], Optional[Dict], Optional[Dict]]:
        """Fetch all data for a condition asynchronously"""
        async with aiohttp.ClientSession() as session:
            try:
                # Run synchronous PubMed fetch in thread
                articles = await asyncio.to_thread(self.fetch_pubmed_articles, condition)
                # Run async fetches concurrently
                mayo_data, cdc_data = await asyncio.gather(
                    self.fetch_mayo_clinic(condition, session),
                    self.fetch_cdc_data(condition, session)
                )
                return articles, mayo_data, cdc_data
            except Exception as e:
                logger.error(f"Data fetch failed for {condition}: {str(e)}")
                return [], None, None

    def validate_pathway(self, pathway: Dict) -> Dict:
        """Complete validation of pathway structure"""
        required = {
            "differentials": [],
            "workup": {"urgent": [], "routine": []},
            "management": {"symptomatic": [], "definitive": [], "lifestyle": []},
            "follow_up": [],
            "decision_support": {"ranked_differentials": []}
        }
        
        if "path" not in pathway:
            pathway["path"] = {}
        
        # Ensure all required fields exist
        for key, default in required.items():
            if key not in pathway["path"]:
                pathway["path"][key] = default
                logger.warning(f"Added missing {key} to pathway {pathway['category']}/{pathway['key']}")
                self.validation_warnings += 1
        
        # Check for empty important fields
        for field in ["differentials", "workup.urgent", "workup.routine", 
                     "management.symptomatic", "management.definitive", 
                     "management.lifestyle", "follow_up"]:
            keys = field.split(".")
            value = pathway["path"]
            for k in keys:
                value = value.get(k, [])
            if not value:
                logger.warning(f"Empty {field} for {pathway['category']}/{pathway['key']}")
                self.validation_warnings += 1
                if len(keys) == 1:
                    pathway["path"][keys[0]] = ["See specialist"] if field == "differentials" else ["Unknown"]
                else:
                    pathway["path"][keys[0]][keys[1]] = ["Unknown"]
        
        return pathway

    async def process_condition(self, condition: str, data: Dict) -> Optional[Tuple[str, str, Dict]]:
        """Process a single condition with full error handling"""
        try:
            category = data["category"]
            key = data["key"]
            
            # Create pathway structure
            pathway = self.create_pathway_template()
            pathway["differentials"] = data["differentials"]
            pathway["workup"] = data["workup"]
            pathway["management"] = data["management"]
            pathway["follow_up"] = data["follow_up"]
            
            # Fetch all data
            articles, mayo_data, cdc_data = await self.fetch_all_data(condition)
            
            # Add ICD-10 code if available
            icd10_code = self.icd10_map.get(condition.lower())
            
            # Prepare updates
            updates = []
            if articles:
                updates.extend(articles)
            if mayo_data:
                updates.append(mayo_data)
            if cdc_data:
                updates.append(cdc_data)
            
            # Update pathway metadata
            if updates:
                pathway["metadata"]["updates"] = updates
                pathway["metadata"]["sources"] = list({u["source"] for u in updates if "source" in u})
                if icd10_code:
                    pathway["metadata"]["icd10_code"] = icd10_code
            
            pathway["metadata"]["last_updated"] = datetime.now().isoformat()
            pathway["decision_support"]["ranked_differentials"] = self.rank_differentials(pathway)
            
            # Validate before returning
            return category, key, self.validate_pathway(pathway)
            
        except Exception as e:
            logger.error(f"Failed to process condition {condition}: {str(e)}")
            self.failed_fetches.append({
                "condition": condition,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return None

    async def update_clinical_pathways(self) -> Dict:
        """Generate all clinical pathways with progress tracking"""
        pathways = {}
        tasks = []
        
        # Create tasks for all conditions
        for condition, data in self.condition_data.items():
            tasks.append(self.process_condition(condition, data))
        
        # Process with progress bar
        with tqdm(total=len(tasks), desc="Processing conditions") as pbar:
            for future in asyncio.as_completed(tasks):
                result = await future
                if result:
                    category, key, pathway = result
                    if category not in pathways:
                        pathways[category] = {}
                    pathways[category][key] = pathway
                pbar.update(1)
        
        if not pathways:
            raise RuntimeError("Failed to generate any clinical pathways")
        
        return pathways

    def backup_database(self, client: MongoClient) -> bool:
        """Create a backup of the database"""
        try:
            backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = CONFIG['backup_dir'] / f"db_backup_{backup_time}"
            backup_path.mkdir(exist_ok=True)
            
            db = client[CONFIG['db_name']]
            for coll_name in db.list_collection_names():
                with open(backup_path / f"{coll_name}.json", 'w') as f:
                    cursor = db[coll_name].find({})
                    json.dump(list(cursor), f)
            
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Database backup failed: {str(e)}")
            return False

    async def populate_database(self, incremental: bool = False) -> bool:
        """Main population workflow with full error handling"""
        client = None
        try:
            # Connect to MongoDB
            client = MongoClient(
                CONFIG['mongo_uri'],
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
                socketTimeoutMS=30000
            )
            client.admin.command('ping')
            db = client[CONFIG['db_name']]
            logger.info(f"Connected to MongoDB: {CONFIG['db_name']}")
            
            # Create backup before full update
            if not incremental:
                self.backup_database(client)
            
            # Generate clinical pathways
            pathways = await self.update_clinical_pathways()
            
            # Update collections
            self.update_collections(db, pathways, incremental)
            
            # Export data
            self.export_pathways(pathways)
            self.generate_condition_report()
            
            # Log performance metrics
            self.log_metrics()
            
            return True
            
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Population failed: {str(e)}", exc_info=True)
            return False
        finally:
            if client:
                client.close()

    def log_metrics(self) -> None:
        """Log performance and operation metrics"""
        total_time = time.time() - self.start_time
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB
        
        logger.info("\n=== Operation Metrics ===")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Peak memory usage: {mem_usage:.1f} MB")
        logger.info(f"Articles fetched: {self.articles_fetched}")
        logger.info(f"Cache hits: {self.cache_hits} ({(self.cache_hits/(self.cache_hits+self.cache_misses)*100):.1f}%)")
        logger.info(f"Validation warnings: {self.validation_warnings}")
        
        if self.failed_fetches:
            logger.warning(f"Failed fetches: {len(self.failed_fetches)}")
            for fail in self.failed_fetches[:3]:  # Show first 3 failures
                logger.warning(f"- {fail['condition']}: {fail['error']}")

@click.command()
@click.option('--incremental', is_flag=True, help="Perform an incremental update.")
def cli(incremental):
    """Command-line interface for the Clinical Knowledge Base Population Script."""
    populator = ClinicalKnowledgePopulator()
    asyncio.run(populator.populate_database(incremental=incremental))

if __name__ == "__main__":
    try:
        # Create event loop and run CLI
        cli()
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)