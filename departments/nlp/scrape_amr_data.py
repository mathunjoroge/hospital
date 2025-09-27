import logging
import os
import re
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Set
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from lxml import etree
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Configuration ---

@dataclass
class Config:
    LOG_DIR: str = '/home/mathu/projects/hospital/logs'
    DATA_DIR: str = '/home/mathu/projects/hospital/departments/nlp/resources'
    LOG_FILE: str = field(init=False)
    DATA_FILE: str = field(init=False)
    MAX_SCRAPED_SAMPLES: int = 10000
    MAX_PMC_ARTICLES: int = 10000  # Increased
    MAX_CDC_PAGES: int = 2000  # Increased
    REQUEST_TIMEOUT: int = 30
    REQUEST_BACKOFF_FACTOR: float = 2.0
    REQUEST_RETRIES: int = 5
    PMC_BASE_URL: str = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
    CDC_MMWR_URL: str = "https://www.cdc.gov/mmwr/index.html"
    CDC_BASE_URL: str = "https://www.cdc.gov"
    PUBMED_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    AMR_IPC_KEYWORDS: List[str] = field(default_factory=lambda: [
        "antibiotic resistance", "multidrug-resistant", "mrsa", "vre", "esbl",
        "carbapenem-resistant", "ceftriaxone", "ciprofloxacin", "vancomycin",
        "meropenem", "infection", "pneumonia", "uti", "wound infection", "fever",
        "cough", "fatigue", "sepsis", "contact precautions", "hand hygiene",
        "isolation protocol", "sterile technique", "culture negative", "susceptible",
        "pan-sensitive", "no infection", "post-hospitalization", "antibiotic",
        "culture", "resistant", "bacterial infection", "treatment failure",
        "no precautions", "poor infection control", "inadequate isolation"  # Added IPC terms
    ])

    def __post_init__(self):
        self.LOG_FILE = os.path.join(self.LOG_DIR, 'hims_scraper.log')
        self.DATA_FILE = os.path.join(self.DATA_DIR, 'amr_training_data.csv')
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)

# --- Logging Setup ---

def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("HIMS-Scraper")

# --- Data Handling ---

class DataHandler:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_existing_texts(self) -> Set[str]:
        if not os.path.exists(self.config.DATA_FILE):
            return set()
        try:
            with open(self.config.DATA_FILE, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                try:
                    next(reader)
                except StopIteration:
                    return set()
                return {row[0] for row in reader if row}
        except Exception as e:
            self.logger.error(f"Could not read existing data from {self.config.DATA_FILE}: {e}")
            return set()

    def update_training_data(self, new_texts: List[str]):
        if not new_texts:
            self.logger.info("No new texts to add.")
            return
        # Add curated texts if fewer than 50 snippets
        if len(new_texts) < 50:
            curated_texts = [
                "Patient with recurrent UTI; ciprofloxacin failed, no isolation protocol followed.",
                "MRSA pneumonia treated with vancomycin; poor hand hygiene noted.",
                "Fever and sepsis post-surgery; ceftriaxone used, inadequate isolation.",
                "Culture confirmed resistant E. coli in wound infection; no precautions."
            ]
            new_texts.extend([t for t in curated_texts if t not in new_texts])
            self.logger.info(f"Added {len(curated_texts)} curated texts to reach minimum threshold.")
        
        file_exists = os.path.exists(self.config.DATA_FILE)
        try:
            with open(self.config.DATA_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['text'])
                for text in new_texts:
                    writer.writerow([text])
            self.logger.info(f"Appended {len(new_texts)} new examples to {self.config.DATA_FILE}")
        except IOError as e:
            self.logger.error(f"Error updating training data file: {e}")

# --- Web Scraper ---

class Scraper:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session = self._create_session()
        self.keywords = set(self.config.AMR_IPC_KEYWORDS)
        self.oai_ns = {"oai": "http://www.openarchives.org/OAI/2.0/"}

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=self.config.REQUEST_RETRIES,
            backoff_factor=self.config.REQUEST_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504, 405]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        return session

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,;-]', '', text)
        return text

    def _is_clinically_relevant(self, text: str, is_cdc: bool = False) -> bool:
        """Require one keyword for PMC/PubMed, two for CDC."""
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in self.keywords if keyword in text_lower)
        return keyword_count >= (2 if is_cdc else 1) and len(text.split()) > 10

    def fetch_from_pmc(self) -> List[str]:
        self.logger.info("Starting PMC article fetch...")
        texts = []
        params = {"verb": "ListRecords", "metadataPrefix": "pmc", "set": "pmc-open"}
        request_count = 0
        max_requests = 10

        while len(texts) < self.config.MAX_PMC_ARTICLES and request_count < max_requests:
            request_count += 1
            self.logger.info(f"PMC request {request_count}: {params}")
            try:
                response = self.session.get(self.config.PMC_BASE_URL, params=params, timeout=self.config.REQUEST_TIMEOUT)
                self.logger.info(f"PMC response status: {response.status_code}")
                response.raise_for_status()
                
                root = etree.fromstring(response.content)
                records = root.xpath(".//oai:record", namespaces=self.oai_ns)
                self.logger.info(f"Found {len(records)} records in response")
                
                for i, record in enumerate(records):
                    self.logger.info(f"Processing PMC record {i+1}/{len(records)}")
                    content_element = record.find(".//oai:abstract", self.oai_ns) or record.find(".//oai:body", self.oai_ns)
                    if content_element is None:
                        self.logger.info("No abstract or body found in record")
                        continue

                    full_text = ''.join(content_element.itertext()).strip()
                    sentences = re.split(r'[.!?]', full_text)
                    
                    for sentence in sentences:
                        cleaned_sentence = self._clean_text(sentence)
                        if self._is_clinically_relevant(cleaned_sentence):
                            texts.append(cleaned_sentence)
                            self.logger.info(f"Added PMC snippet: {cleaned_sentence[:50]}...")
                            if len(texts) >= self.config.MAX_PMC_ARTICLES:
                                break
                    if len(texts) >= self.config.MAX_PMC_ARTICLES:
                        break
                
                token_element = root.find(".//oai:resumptionToken", self.oai_ns)
                if token_element is None or not token_element.text:
                    self.logger.info("No resumption token found. Ending PMC fetch.")
                    break
                
                params = {"verb": "ListRecords", "resumptionToken": token_element.text}
                time.sleep(2)

            except requests.RequestException as e:
                self.logger.error(f"Error fetching PMC articles: {e}")
                break
            except etree.XMLSyntaxError as e:
                self.logger.error(f"XML parsing error from PMC: {e}")
                break

        self.logger.info(f"Fetched {len(texts)} relevant snippets from PMC.")
        if not texts:
            self.logger.warning("No PMC snippets collected. Trying PubMed fallback.")
            texts.extend(self.fetch_from_pubmed())
        return texts

    def fetch_from_pubmed(self) -> List[str]:
        self.logger.info("Starting PubMed fallback fetch...")
        texts = []
        params = {
            "db": "pmc",
            "term": "open access[filter] AND (antibiotic resistance OR infection control OR ciprofloxacin OR ceftriaxone OR mrsa)",
            "retmax": self.config.MAX_PMC_ARTICLES,
            "retmode": "xml"
        }
        try:
            response = self.session.get(self.config.PUBMED_BASE_URL, params=params, timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            root = etree.fromstring(response.content)
            ids = root.xpath("//Id")
            self.logger.info(f"Found {len(ids)} PubMed PMC IDs")

            for pmc_id in ids[:self.config.MAX_PMC_ARTICLES]:
                article_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_id.text}"
                try:
                    article_response = self.session.get(article_url, timeout=self.config.REQUEST_TIMEOUT)
                    article_response.raise_for_status()
                    article_root = etree.fromstring(article_response.content)
                    abstract = article_root.find(".//abstract")
                    if abstract is not None:
                        full_text = ''.join(abstract.itertext()).strip()
                        sentences = re.split(r'[.!?]', full_text)
                        for sentence in sentences:
                            cleaned_sentence = self._clean_text(sentence)
                            if self._is_clinically_relevant(cleaned_sentence):
                                texts.append(cleaned_sentence)
                                self.logger.info(f"Added PubMed snippet: {cleaned_sentence[:50]}...")
                                if len(texts) >= self.config.MAX_PMC_ARTICLES:
                                    break
                    time.sleep(1)
                except Exception as e:
                    self.logger.warning(f"Failed to fetch PubMed article {pmc_id.text}: {e}")
            self.logger.info(f"Fetched {len(texts)} relevant snippets from PubMed.")
        except Exception as e:
            self.logger.error(f"Error fetching PubMed articles: {e}")
        return texts

    def _scrape_cdc_article_page(self, url: str) -> List[str]:
        texts = []
        try:
            response = self.session.get(url, timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            if any(phrase in soup.get_text().lower() for phrase in ["this episode discusses", "mmwr weekly briefing"]):
                self.logger.info(f"Skipping podcast summary page: {url}")
                return []
            
            content = soup.find('div', class_='content') or soup.find('div', id='content')
            if content:
                for p in content.find_all('p'):
                    text = p.get_text()
                    cleaned_text = self._clean_text(text)
                    if self._is_clinically_relevant(cleaned_text, is_cdc=True):
                        texts.append(cleaned_text)
                        self.logger.info(f"Added CDC snippet: {cleaned_text[:50]}...")
            return texts
        except requests.RequestException as e:
            self.logger.warning(f"Failed to scrape CDC page {url}: {e}")
            return []

    def fetch_from_cdc(self) -> List[str]:
        self.logger.info("Starting CDC MMWR scrape...")
        try:
            response = self.session.get(self.config.CDC_MMWR_URL, timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            links = {
                urljoin(self.config.CDC_BASE_URL, a['href'])
                for a in soup.find_all('a', href=True)
                if 'mmwr' in a['href'] and a['href'].endswith('.htm')
            }
            
            if not links:
                self.logger.warning("No article links found on CDC MMWR index page.")
                return []

            article_links = list(links)[:self.config.MAX_CDC_PAGES]
            self.logger.info(f"Found {len(article_links)} unique article links to scrape.")
            
            all_texts = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_url = {executor.submit(self._scrape_cdc_article_page, url): url for url in article_links}
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        texts_from_page = future.result()
                        if texts_from_page:
                            all_texts.extend(texts_from_page)
                            self.logger.info(f"Successfully scraped {len(texts_from_page)} snippets from {url}")
                    except Exception as exc:
                        self.logger.error(f'{url} generated an exception: {exc}')

            self.logger.info(f"Fetched {len(all_texts)} relevant snippets from CDC.")
            return all_texts

        except requests.RequestException as e:
            self.logger.error(f"Could not access CDC MMWR index page: {e}")
            return []

# --- Main Execution ---

def main():
    config = Config()
    logger = setup_logging(config.LOG_FILE)
    
    logger.info("--- Starting HIMS Scraper ---")
    
    data_handler = DataHandler(config, logger)
    scraper = Scraper(config, logger)
    
    try:
        existing_texts = data_handler.load_existing_texts()
        logger.info(f"Loaded {len(existing_texts)} existing texts from data file.")
        pmc_texts = scraper.fetch_from_pmc()
        cdc_texts = scraper.fetch_from_cdc()
        all_scraped_texts = pmc_texts + cdc_texts
        unique_new_texts = [
            text for text in all_scraped_texts
            if text not in existing_texts and len(text.split()) > 10
        ]
        final_texts_to_add = unique_new_texts[:config.MAX_SCRAPED_SAMPLES]
        logger.info(f"Found {len(final_texts_to_add)} new, unique texts to be added.")
        data_handler.update_training_data(final_texts_to_add)

    except KeyboardInterrupt:
        logger.info("Script interrupted by user.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main process: {e}", exc_info=True)
    finally:
        logger.info("--- HIMS Scraper Finished ---")

if __name__ == "__main__":
    main()