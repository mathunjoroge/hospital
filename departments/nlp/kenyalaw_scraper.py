import json
import logging
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Set, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse
import io

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------------------------------------------------------- #
#                               CONFIGURATION                                 #
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    BASE_PROJECT_DIR: str = os.path.join(os.path.expanduser("~"), "projects", "kenya_law")

    # Directories and files - MATCHING TRAINER EXPECTATIONS
    LOG_DIR: str = field(init=False)
    DATA_DIR: str = field(init=False)
    
    # Output files - EXACTLY WHAT THE TRAINER EXPECTS
    CONSTITUTION_FILE: str = field(init=False)
    ACTS_FILE: str = field(init=False)
    COUNTIES_FILE: str = field(init=False)
    TRAINING_DATA_FILE: str = field(init=False)  # This is the key file for trainer
    
    # Scraping limits
    MAX_CASES: int = 2000  # Matches trainer's limit
    MAX_ACTS: int = 500
    MAX_COUNTY_LAWS: int = 100
    REQUEST_TIMEOUT: int = 30
    
    # URLs that actually work
    BASE_URL: str = "https://kenyalaw.org"
    NEW_BASE_URL: str = "https://new.kenyalaw.org"
    LEGISLATION_HOME: str = "https://new.kenyalaw.org/legislation/"
    JUDGMENTS_URL: str = "https://new.kenyalaw.org/judgments/"
    COUNTIES_URL: str = "https://new.kenyalaw.org/legislation/counties"
    
    # Technical settings
    MAX_SCRAPE_WORKERS: int = 5

    def __post_init__(self) -> None:
        # Directory setup
        self.LOG_DIR = os.path.join(self.BASE_PROJECT_DIR, "logs")
        self.DATA_DIR = os.path.join(self.BASE_PROJECT_DIR, "data")
        
        # Create directories
        for d in [self.LOG_DIR, self.DATA_DIR]:
            os.makedirs(d, exist_ok=True)
            
        # File paths - EXACTLY WHAT TRAINER EXPECTS
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.LOG_FILE = os.path.join(self.LOG_DIR, f"kenyalaw_scraper_{timestamp}.log")
        self.CONSTITUTION_FILE = os.path.join(self.DATA_DIR, "constitution.json")
        self.ACTS_FILE = os.path.join(self.DATA_DIR, "acts_of_kenya.json")
        self.COUNTIES_FILE = os.path.join(self.DATA_DIR, "county_legislation.json")
        self.TRAINING_DATA_FILE = os.path.join(self.DATA_DIR, "kenya_law_training_data.jsonl")  # Key file!

# --------------------------------------------------------------------------- #
#                                 LOGGING                                    #
# --------------------------------------------------------------------------- #

def setup_logging(log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger("KenyaLaw-Scraper-Trainer-Compatible")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_file, encoding="utf-8")
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# --------------------------------------------------------------------------- #
#                          UNIVERSAL SCRAPING TOOLS                          #
# --------------------------------------------------------------------------- #

class LawScraper:
    def __init__(self, cfg: Config, log: logging.Logger):
        self.cfg = cfg
        self.log = log
        self.session = self._create_session()

    def _create_session(self):
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br"
        })
        return s

    def extract_law_content(self, url: str) -> Tuple[Optional[str], Dict]:
        """Extract content from any law URL with metadata"""
        metadata = {
            'url': url,
            'scraped_at': datetime.now().isoformat(),
            'title': '',
            'word_count': 0
        }
        
        try:
            # Handle HTML content
            resp = self.session.get(url, timeout=self.cfg.REQUEST_TIMEOUT)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            # Extract title
            title_elem = soup.find("h1") or soup.find("title")
            metadata['title'] = title_elem.get_text(strip=True) if title_elem else "Unknown Law"

            # Extract HTML content - try multiple selectors
            content_selectors = [
                "div.act-content",
                "div.fr-view",
                "div.content",
                "article",
                "main",
                ".law-content",
                ".document-content"
            ]

            content = None
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    break

            if not content:
                content = soup.find('main') or soup.find('article') or soup.find('body')

            if content:
                # Clean up content
                for element in content.select("script, style, nav, header, footer, .nav, .header, .footer, .tools"):
                    element.decompose()

                # Extract text
                text = content.get_text(separator="\n", strip=True)
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                metadata['word_count'] = len(text.split())
                
                return text, metadata

        except Exception as e:
            self.log.error(f"Failed to extract content from {url}: {e}")
            
        return None, metadata

    def find_akn_links(self, base_url: str) -> List[Tuple[str, str]]:
        """Find all AKN (Akoma Ntoso) law links on a page"""
        laws = []
        try:
            resp = self.session.get(base_url, timeout=self.cfg.REQUEST_TIMEOUT)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            
            # Find all AKN links
            akn_links = soup.find_all('a', href=re.compile(r'/akn/ke/'))
            for link in akn_links:
                title = link.get_text(strip=True)
                href = link.get('href')
                if href and title and len(title) > 5:
                    full_url = urljoin(self.cfg.NEW_BASE_URL, href)
                    laws.append((title, full_url))
            
        except Exception as e:
            self.log.error(f"Failed to find AKN links on {base_url}: {e}")
            
        return laws

# --------------------------------------------------------------------------- #
#                         CONSTITUTION SCRAPER                               #
# --------------------------------------------------------------------------- #

def scrape_constitution(cfg: Config, log: logging.Logger) -> Dict:
    """Scrape the Constitution of Kenya - IN TRAINER-COMPATIBLE FORMAT"""
    log.info("Scraping Constitution of Kenya...")
    
    if os.path.exists(cfg.CONSTITUTION_FILE):
        log.info("Constitution already exists. Loading from file.")
        with open(cfg.CONSTITUTION_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    scraper = LawScraper(cfg, log)
    
    # Try constitution sources
    sources = [
        "https://new.kenyalaw.org/akn/ke/act/2010/constitution/eng@2010-09-03",
        "https://new.kenyalaw.org/akn/ke/act/2010/constitution",
    ]
    
    constitution_data = {}
    
    for source in sources:
        try:
            content, metadata = scraper.extract_law_content(source)
            if content and len(content.split()) > 1000:
                # FORMAT FOR TRAINER: Simple title: content structure
                constitution_data = {
                    'Constitution of Kenya': content,
                    'Preamble': extract_preamble(content),
                    'Bill of Rights': extract_bill_of_rights(content)
                }
                break
        except Exception as e:
            log.warning(f"Constitution source {source} failed: {e}")
            continue
    
    # Save constitution data
    if constitution_data:
        with open(cfg.CONSTITUTION_FILE, 'w', encoding='utf-8') as f:
            json.dump(constitution_data, f, ensure_ascii=False, indent=2)
        log.info(f"Constitution saved with {len(constitution_data)} sections")
    else:
        log.error("Failed to scrape constitution")
        
    return constitution_data

def extract_preamble(content: str) -> str:
    """Extract preamble from constitution content"""
    if not isinstance(content, str):
        return "Preamble content not found"
    
    lines = content.split('\n')
    preamble_lines = []
    in_preamble = False
    
    for line in lines:
        line = line.strip()
        if 'preamble' in line.lower() or 'we the people' in line.lower():
            in_preamble = True
        if in_preamble and line and not line.lower().startswith('chapter'):
            preamble_lines.append(line)
        elif in_preamble and line.lower().startswith('chapter'):
            break
    
    return ' '.join(preamble_lines) if preamble_lines else "Preamble content not found"

def extract_bill_of_rights(content: str) -> str:
    """Extract bill of rights from constitution content"""
    if not isinstance(content, str):
        return "Bill of Rights content not found"
    
    lines = content.split('\n')
    rights_lines = []
    in_rights = False
    
    for line in lines:
        line = line.strip()
        if 'bill of rights' in line.lower() or 'chapter four' in line.lower():
            in_rights = True
        if in_rights and line:
            rights_lines.append(line)
        elif in_rights and 'chapter five' in line.lower():
            break
    
    return ' '.join(rights_lines) if rights_lines else "Bill of Rights content not found"

# --------------------------------------------------------------------------- #
#                           ACTS OF KENYA SCRAPER                            #
# --------------------------------------------------------------------------- #

def scrape_acts_of_kenya(cfg: Config, log: logging.Logger) -> Dict[str, str]:
    """Scrape Acts of Kenya - IN TRAINER-COMPATIBLE FORMAT"""
    log.info("Scraping Acts of Kenya...")
    
    if os.path.exists(cfg.ACTS_FILE):
        log.info("Acts of Kenya already exist. Loading from file.")
        with open(cfg.ACTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    scraper = LawScraper(cfg, log)
    acts_data = {}
    
    try:
        # Find acts from legislation home
        laws = scraper.find_akn_links(cfg.LEGISLATION_HOME)
        acts = [law for law in laws if '/akn/ke/act/' in law[1]]
        
        log.info(f"Found {len(acts)} potential acts")
        
        # Process each act
        for i, (title, url) in enumerate(acts):
            if i >= cfg.MAX_ACTS:
                break
                
            log.info(f"Processing act {i+1}/{len(acts)}: {title}")
            
            content, metadata = scraper.extract_law_content(url)
            if content and len(content.split()) > 100:
                # FORMAT FOR TRAINER: Simple title: content structure
                acts_data[title] = content
                log.info(f"  ✓ {title} ({len(content.split())} words)")
                
            time.sleep(1)
            
    except Exception as e:
        log.error(f"Acts scraping failed: {e}")
    
    # Save acts data
    with open(cfg.ACTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(acts_data, f, ensure_ascii=False, indent=2)
    
    log.info(f"Acts of Kenya scraping completed: {len(acts_data)} acts saved")
    return acts_data

# --------------------------------------------------------------------------- #
#                          COUNTY LEGISLATION SCRAPER                        #
# --------------------------------------------------------------------------- #

def scrape_county_legislation(cfg: Config, log: logging.Logger) -> Dict[str, Dict]:
    """Scrape county legislation - IN TRAINER-COMPATIBLE FORMAT"""
    log.info("Scraping County Legislation...")
    
    if os.path.exists(cfg.COUNTIES_FILE):
        log.info("County legislation already exists. Loading from file.")
        with open(cfg.COUNTIES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    scraper = LawScraper(cfg, log)
    counties_data = {}
    
    try:
        # Get main counties page
        resp = scraper.session.get(cfg.COUNTIES_URL, timeout=cfg.REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        
        # Extract county links
        county_links = []
        county_elements = soup.select('a[href*="/legislation/county/"]')
        
        for element in county_elements:
            county_name = element.get_text(strip=True)
            href = element.get('href')
            if county_name and href and 'county' in county_name.lower():
                full_url = urljoin(cfg.NEW_BASE_URL, href)
                county_links.append((county_name, full_url))
        
        log.info(f"Found {len(county_links)} counties")
        
        # Process each county (limit for testing)
        for county_name, county_url in county_links[:5]:
            try:
                log.info(f"Processing county: {county_name}")
                county_laws = {}
                
                # Find laws in county page
                laws = scraper.find_akn_links(county_url)
                
                for law_title, law_url in laws[:3]:  # Limit laws per county
                    content, metadata = scraper.extract_law_content(law_url)
                    if content and len(content.split()) > 50:
                        # FORMAT FOR TRAINER: Nested county structure with laws
                        county_laws[law_title] = {
                            'content': content,
                            'url': law_url,
                            'word_count': len(content.split())
                        }
                
                if county_laws:
                    counties_data[county_name] = {
                        'laws': county_laws,
                        'total_laws': len(county_laws)
                    }
                    log.info(f"  ✓ {county_name}: {len(county_laws)} laws")
                    
                time.sleep(2)
                
            except Exception as e:
                log.error(f"Failed to process county {county_name}: {e}")
                continue
                
    except Exception as e:
        log.error(f"County legislation scraping failed: {e}")
    
    # Save counties data
    with open(cfg.COUNTIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(counties_data, f, ensure_ascii=False, indent=2)
    
    log.info(f"County legislation scraping completed: {len(counties_data)} counties saved")
    return counties_data

# --------------------------------------------------------------------------- #
#                            CASE LAW SCRAPER                                #
# --------------------------------------------------------------------------- #

def scrape_case_law(cfg: Config, log: logging.Logger) -> List[Dict]:
    """Scrape case law - IN TRAINER-COMPATIBLE FORMAT"""
    log.info("Scraping Case Law...")
    
    case_data = []
    
    try:
        scraper = LawScraper(cfg, log)
        
        # Get judgments page
        laws = scraper.find_akn_links(cfg.JUDGMENTS_URL)
        judgment_laws = [law for law in laws if '/judgment/' in law[1]]
        
        log.info(f"Found {len(judgment_laws)} judgment URLs")
        
        # Process each judgment
        for i, (title, url) in enumerate(judgment_laws[:cfg.MAX_CASES]):
            try:
                log.info(f"Processing judgment {i+1}/{len(judgment_laws)}: {title}")
                
                content, metadata = scraper.extract_law_content(url)
                if content and len(content.split()) > 200:
                    # FORMAT FOR TRAINER: Exact structure expected
                    case_info = {
                        "case_id": f"case_{i}",
                        "case_name": title,
                        "url": url,
                        "text": content,
                        "metadata": {
                            "court": extract_court_from_title(title),
                            "date": extract_date_from_title(title),
                            "case_number": extract_case_number(title)
                        },
                        "scraped_at": datetime.now().isoformat()
                    }
                    
                    case_data.append(case_info)
                    log.info(f"  ✓ {title} ({len(content.split())} words)")
                    
                time.sleep(1)
                
            except Exception as e:
                log.error(f"Failed to process judgment {url}: {e}")
                continue
                
    except Exception as e:
        log.error(f"Case law scraping failed: {e}")
    
    # Save to JSONL file - EXACTLY WHAT TRAINER EXPECTS
    if case_data:
        with open(cfg.TRAINING_DATA_FILE, 'w', encoding='utf-8') as f:
            for case in case_data:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        log.info(f"Case law saved to {cfg.TRAINING_DATA_FILE}: {len(case_data)} cases")
    
    return case_data

def extract_court_from_title(title: str) -> str:
    """Extract court name from case title"""
    if not isinstance(title, str):
        return "Kenyan Court"
    
    courts = ['Supreme Court', 'Court of Appeal', 'High Court', 'Magistrate Court', 
              'Employment Court', 'Environment Court', 'ELC', 'KECA', 'KEHC']
    for court in courts:
        if court in title:
            return court
    return "Kenyan Court"

def extract_date_from_title(title: str) -> str:
    """Extract date from case title"""
    if not isinstance(title, str):
        return "Unknown date"
    
    date_pattern = r'(\d{1,2}\s+\w+\s+\d{4})'
    match = re.search(date_pattern, title)
    return match.group(1) if match else "Unknown date"

def extract_case_number(title: str) -> str:
    """Extract case number from title"""
    if not isinstance(title, str):
        return "Unknown case number"
    
    case_pattern = r'[A-Za-z]+\s+[A-Za-z]+\s+[E\d]+\s+of\s+\d{4}'
    match = re.search(case_pattern, title)
    return match.group(0) if match else "Unknown case number"

# --------------------------------------------------------------------------- #
#                      SAFE WORD COUNT CALCULATION                           #
# --------------------------------------------------------------------------- #

def safe_word_count(data: any) -> int:
    """Safely calculate word count for any data type"""
    if isinstance(data, str):
        return len(data.split())
    elif isinstance(data, dict):
        # For dictionaries, sum word counts of all string values
        total = 0
        for value in data.values():
            if isinstance(value, str):
                total += len(value.split())
        return total
    elif isinstance(data, list):
        # For lists, sum word counts of all string items
        total = 0
        for item in data:
            if isinstance(item, str):
                total += len(item.split())
            elif isinstance(item, dict):
                total += safe_word_count(item)
        return total
    else:
        return 0

# --------------------------------------------------------------------------- #
#                            MAIN SCRAPER CLASS                              #
# --------------------------------------------------------------------------- #

class KenyaLawScraper:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log = setup_logging(cfg.LOG_FILE)

    def scrape_all_laws(self) -> bool:
        """Scrape all laws in trainer-compatible format"""
        self.log.info("=== STARTING KENYA LAW SCRAPING (TRAINER-COMPATIBLE) ===")
        
        success = True
        
        try:
            # 1. Constitution - creates constitution.json
            self.log.info("\n--- SCRAPING CONSTITUTION ---")
            constitution_data = scrape_constitution(self.cfg, self.log)
            
            # 2. Acts of Kenya - creates acts_of_kenya.json  
            self.log.info("\n--- SCRAPING ACTS OF KENYA ---")
            acts_data = scrape_acts_of_kenya(self.cfg, self.log)
            
            # 3. County Legislation - creates county_legislation.json
            self.log.info("\n--- SCRAPING COUNTY LEGISLATION ---")
            counties_data = scrape_county_legislation(self.cfg, self.log)
            
            # 4. Case Law - creates kenya_law_training_data.jsonl (KEY FILE!)
            self.log.info("\n--- SCRAPING CASE LAW ---")
            case_data = scrape_case_law(self.cfg, self.log)
            
            # Print summary
            self._print_scraping_summary(constitution_data, acts_data, counties_data, case_data)
            
        except Exception as e:
            self.log.error(f"Scraping failed: {e}", exc_info=True)
            success = False
            
        return success

    def _print_scraping_summary(self, constitution: Dict, acts: Dict, counties: Dict, cases: List):
        """Print a summary of scraping results"""
        self.log.info("\n" + "="*60)
        self.log.info("KENYA LAW SCRAPING SUMMARY (TRAINER-COMPATIBLE)")
        self.log.info("="*60)
        
        self.log.info(f"  Constitution sections: {len(constitution) if constitution else 0}")
        self.log.info(f"  Acts of Kenya: {len(acts) if acts else 0}")
        self.log.info(f"  Counties with laws: {len(counties) if counties else 0}")
        self.log.info(f"  Case Law judgments: {len(cases)}")
        
        # Use safe word count calculation
        total_words = 0
        if constitution:
            total_words += safe_word_count(constitution)
        if acts:
            total_words += safe_word_count(acts)
        if cases:
            total_words += safe_word_count(cases)
        
        self.log.info(f"  Total training words: {total_words:,}")
        
        self.log.info("-"*60)
        self.log.info(f"  KEY FILES CREATED:")
        self.log.info(f"  • {self.cfg.CONSTITUTION_FILE}")
        self.log.info(f"  • {self.cfg.ACTS_FILE}")
        self.log.info(f"  • {self.cfg.COUNTIES_FILE}")
        self.log.info(f"  • {self.cfg.TRAINING_DATA_FILE} (for trainer)")
        self.log.info("="*60)

# --------------------------------------------------------------------------- #
#                               MAIN EXECUTION                               #
# --------------------------------------------------------------------------- #

def main():
    """Main execution function"""
    cfg = Config()
    
    # Create scraper instance
    scraper = KenyaLawScraper(cfg)
    
    # Run scraping
    success = scraper.scrape_all_laws()
    
    if success:
        print(f"\n✅ Scraping completed successfully!")
        print(f"📁 Data saved to: {cfg.DATA_DIR}")
        print(f"📄 Key file for trainer: {cfg.TRAINING_DATA_FILE}")
        
        # Check what files were created
        data_files = os.listdir(cfg.DATA_DIR)
        print(f"📋 Files created: {', '.join(data_files)}")
    else:
        print(f"\n❌ Scraping failed. Check logs: {cfg.LOG_FILE}")

if __name__ == "__main__":
    main()