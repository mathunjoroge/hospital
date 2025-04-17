import requests
from bs4 import BeautifulSoup
import pdfplumber
import json
import re
import os
from urllib.parse import urljoin
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from Bio import Entrez  # For PubMed API

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Email for PubMed API
Entrez.email = "your.email@example.com"  # Replace with your email

# Base URLs for scraping
SOURCES = [
    {
        "name": "Nationwide Children's",
        "url": "https://www.nationwidechildrens.org/conditions",
        "fallbacks": [
            "https://www.nationwidechildrens.org/for-medical-professionals",
            "https://www.nationwidechildrens.org/for-medical-professionals/tools-for-providers"
        ]
    },
    {
        "name": "NICE",
        "url": "https://www.nice.org.uk/guidance/conditions-and-diseases",
        "fallbacks": [
            "https://www.nice.org.uk/guidance",
            "https://www.nice.org.uk/guidance/published"
        ]
    },
    {
        "name": "Mayo Clinic",
        "url": "https://www.mayoclinic.org/diseases-conditions",
        "fallbacks": [
            "https://www.mayoclinic.org/diseases-conditions/index?letter=A",
            "https://www.mayoclinic.org/diseases-conditions/index?letter=Z"
        ]
    },
    {
        "name": "CDC",
        "url": "https://www.cdc.gov/diseases-conditions/",
        "fallbacks": [
            "https://www.cdc.gov/az/",
            "https://www.cdc.gov/health-topics.html"
        ]
    },
    {
        "name": "WHO",
        "url": "https://www.who.int/health-topics/",
        "fallbacks": [
            "https://www.who.int/health-topics/#A",
            "https://www.who.int/health-topics/#Z"
        ]
    },
    {
        "name": "PubMed",
        "url": "https://pubmed.ncbi.nlm.nih.gov/",
        "api": True,  # Indicates API-based source
        "fallbacks": []
    },
    {
        "name": "UpToDate",
        "url": "https://www.uptodate.com/contents/search",
        "fallbacks": [
            "https://www.uptodate.com/contents/table-of-contents"
        ]
    },
    {
        "name": "ClinicalKey",
        "url": "https://www.clinicalkey.com/#!/browse/conditions",
        "fallbacks": [
            "https://www.clinicalkey.com/#!/browse/guidelines"
        ]
    }
]

# Output JSON file
OUTPUT_FILE = "/home/mathu/projects/hospital/departments/nlp/knowledge_base/clinical_pathways.json"

# Headers for requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def fetch_page(url, use_selenium=False):
    """Fetch webpage content with requests or Selenium."""
    if use_selenium:
        try:
            options = Options()
            options.add_argument("--headless")
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            time.sleep(5)
            html = driver.page_source
            driver.quit()
            logging.info(f"Successfully fetched {url} with Selenium")
            return html
        except Exception as e:
            logging.error(f"Failed to fetch {url} with Selenium: {e}")
            return None
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        logging.info(f"Successfully fetched {url}")
        return response.text
    except requests.RequestException as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

def fetch_pubmed_articles(symptom):
    """Fetch relevant PubMed articles for a symptom or condition."""
    try:
        query = f"{symptom} clinical guidelines OR clinical pathways OR differential diagnosis"
        handle = Entrez.esearch(db="pubmed", term=query, retmax=5, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        
        if not record["IdList"]:
            logging.warning(f"No PubMed articles found for {symptom}")
            return []
        
        handle = Entrez.efetch(db="pubmed", id=record["IdList"], retmode="xml")
        articles = Entrez.read(handle)
        handle.close()
        
        results = []
        for article in articles["PubmedArticle"]:
            try:
                title = article["MedlineCitation"]["Article"]["ArticleTitle"]
                abstract = ""
                if "Abstract" in article["MedlineCitation"]["Article"]:
                    abstract = " ".join(article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"])
                results.append({"title": title, "abstract": abstract.lower()})
            except KeyError:
                continue
        logging.info(f"Fetched {len(results)} PubMed articles for {symptom}")
        return results
    except Exception as e:
        logging.error(f"Failed to fetch PubMed articles for {symptom}: {e}")
        return []

def download_pdf(url, filename):
    """Download PDF file."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        with open(filename, "wb") as f:
            f.write(response.content)
        logging.info(f"Downloaded PDF: {filename}")
        return True
    except requests.RequestException as e:
        logging.error(f"Failed to download PDF {url}: {e}")
        return False

def extract_pdf_content(pdf_path, source_name):
    """Extract text from PDF and parse into pathway structure."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages).lower()
        
        pathway = {"differentials": [], "workup": {"urgent": [], "routine": []}, "management": {"symptomatic": [], "definitive": []}}
        
        # Source-specific regex
        diff_patterns = {
            "Mayo Clinic": r"(?:causes include|possible causes|consider|may include)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "CDC": r"(?:caused by|risk factors|may include)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "WHO": r"(?:causes|etiologies|may include)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "PubMed": r"(?:differential diagnosis|possible etiologies|consider|causes include)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "UpToDate": r"(?:differential diagnosis|etiologies|may include)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "ClinicalKey": r"(?:causes|differential|consider)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "default": r"(?:differential diagnosis|possible causes|consider|may include|causes include)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
        }
        urgent_patterns = {
            "Mayo Clinic": r"(?:seek immediate|emergency care|urgent|prompt)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "CDC": r"(?:seek care immediately|urgent|what to do if)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "WHO": r"(?:urgent|immediate action|emergency)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "PubMed": r"(?:urgent evaluation|emergency|immediate)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "UpToDate": r"(?:urgent|immediate|prompt)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "ClinicalKey": r"(?:emergency|urgent care|prompt)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "default": r"(?:urgent|immediate|initial evaluation|emergency|prompt)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
        }
        routine_patterns = {
            "Mayo Clinic": r"(?:diagnosis includes|tests include|diagnostic|follow-up)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "CDC": r"(?:diagnosed by|tests for|routine)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "WHO": r"(?:diagnostic|investigation|assessment)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "PubMed": r"(?:diagnostic workup|routine tests|investigations)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "UpToDate": r"(?:diagnostic evaluation|tests|assessment)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "ClinicalKey": r"(?:diagnostic|tests|workup)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "default": r"(?:routine|diagnostic|follow-up|investigation|assessment|tests include)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
        }
        symp_patterns = {
            "Mayo Clinic": r"(?:relieve symptoms|symptom relief|initial treatment)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "CDC": r"(?:manage symptoms|symptom management|relief)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "WHO": r"(?:symptomatic treatment|supportive care)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "PubMed": r"(?:symptomatic management|supportive care|relief)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "UpToDate": r"(?:symptom management|supportive treatment)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "ClinicalKey": r"(?:symptomatic care|relief|initial treatment)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "default": r"(?:symptomatic|supportive|initial treatment|relief|symptom management)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
        }
        def_patterns = {
            "Mayo Clinic": r"(?:treatment includes|long-term care|specific treatment)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "CDC": r"(?:treatment for|cure|management includes)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "WHO": r"(?:guideline|definitive treatment|long-term management)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "PubMed": r"(?:definitive treatment|therapeutic guidelines|management)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "UpToDate": r"(?:treatment guidelines|definitive care|management)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "ClinicalKey": r"(?:definitive treatment|management|therapy)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
            "default": r"(?:definitive|specific|long-term|curative|guideline|treatment includes)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
        }
        
        # Extract differentials
        diff_pattern = diff_patterns.get(source_name, diff_patterns["default"])
        diff_matches = re.findall(diff_pattern, text, re.IGNORECASE)
        pathway["differentials"] = [d.strip() for match in diff_matches for d in match.split(",") if d.strip() and len(d.strip()) > 3 and "condition" not in d.lower()][:4]
        
        # Extract workup
        urgent_pattern = urgent_patterns.get(source_name, urgent_patterns["default"])
        routine_pattern = routine_patterns.get(source_name, routine_patterns["default"])
        urgent_matches = re.findall(urgent_pattern, text, re.IGNORECASE)
        routine_matches = re.findall(routine_pattern, text, re.IGNORECASE)
        pathway["workup"]["urgent"] = [u.strip() for match in urgent_matches for u in match.split(",") if u.strip() and len(u.strip()) > 5 and "call" not in u.lower()][:3]
        pathway["workup"]["routine"] = [r.strip() for match in routine_matches for r in match.split(",") if r.strip() and len(r.strip()) > 5 and "learn" not in r.lower()][:4]
        
        # Extract management
        symp_pattern = symp_patterns.get(source_name, symp_patterns["default"])
        def_pattern = def_patterns.get(source_name, def_patterns["default"])
        symp_matches = re.findall(symp_pattern, text, re.IGNORECASE)
        def_matches = re.findall(def_pattern, text, re.IGNORECASE)
        pathway["management"]["symptomatic"] = [s.strip() for match in symp_matches for s in match.split(",") if s.strip() and len(s.strip()) > 5 and "lifestyle" not in s.lower()][:3]
        pathway["management"]["definitive"] = [d.strip() for match in def_matches for d in match.split(",") if d.strip() and len(d.strip()) > 5 and "research" not in d.lower()][:3]
        
        # Validate content
        if not pathway["differentials"] and not any(pathway["workup"].values()) and not any(pathway["management"].values()):
            logging.warning(f"No structured data extracted from {pdf_path}")
            return None
        return pathway
    except Exception as e:
        logging.error(f"Failed to extract PDF {pdf_path}: {e}")
        return None

def extract_pubmed_content(articles, source_name):
    """Extract pathway data from PubMed article abstracts."""
    pathway = {"differentials": [], "workup": {"urgent": [], "routine": []}, "management": {"symptomatic": [], "definitive": []}}
    
    diff_pattern = r"(?:differential diagnosis|possible etiologies|consider|causes include)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
    urgent_pattern = r"(?:urgent evaluation|emergency|immediate)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
    routine_pattern = r"(?:diagnostic workup|routine tests|investigations)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
    symp_pattern = r"(?:symptomatic management|supportive care|relief)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
    def_pattern = r"(?:definitive treatment|therapeutic guidelines|management)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
    
    for article in articles:
        text = article["abstract"]
        # Extract differentials
        diff_matches = re.findall(diff_pattern, text, re.IGNORECASE)
        pathway["differentials"].extend([d.strip() for match in diff_matches for d in match.split(",") if d.strip() and len(d.strip()) > 3 and "condition" not in d.lower()])
        
        # Extract workup
        urgent_matches = re.findall(urgent_pattern, text, re.IGNORECASE)
        routine_matches = re.findall(routine_pattern, text, re.IGNORECASE)
        pathway["workup"]["urgent"].extend([u.strip() for match in urgent_matches for u in match.split(",") if u.strip() and len(u.strip()) > 5 and "call" not in u.lower()])
        pathway["workup"]["routine"].extend([r.strip() for match in routine_matches for r in match.split(",") if r.strip() and len(r.strip()) > 5 and "learn" not in r.lower()])
        
        # Extract management
        symp_matches = re.findall(symp_pattern, text, re.IGNORECASE)
        def_matches = re.findall(def_pattern, text, re.IGNORECASE)
        pathway["management"]["symptomatic"].extend([s.strip() for match in symp_matches for s in match.split(",") if s.strip() and len(s.strip()) > 5 and "lifestyle" not in s.lower()])
        pathway["management"]["definitive"].extend([d.strip() for match in def_matches for d in match.split(",") if d.strip() and len(d.strip()) > 5 and "research" not in d.lower()])
    
    # Deduplicate and limit
    pathway["differentials"] = list(set(pathway["differentials"]))[:4]
    pathway["workup"]["urgent"] = list(set(pathway["workup"]["urgent"]))[:3]
    pathway["workup"]["routine"] = list(set(pathway["workup"]["routine"]))[:4]
    pathway["management"]["symptomatic"] = list(set(pathway["management"]["symptomatic"]))[:3]
    pathway["management"]["definitive"] = list(set(pathway["management"]["definitive"]))[:3]
    
    if not pathway["differentials"] and not any(pathway["workup"].values()) and not any(pathway["management"].values()):
        logging.warning("No structured data extracted from PubMed articles")
        return None
    return pathway

def categorize_symptom(symptom):
    """Map symptom or condition to a symptom-based category and subcategory."""
    symptom_mappings = {
        "headache|migraine|head pain": {"category": "pain", "subcategory": "head"},
        "chest pain|angina|heart attack": {"category": "pain", "subcategory": "chest"},
        "abdominal pain|abdomen|appendicitis|gerd|cholecystitis|peptic ulcer": {"category": "pain", "subcategory": "abdomen"},
        "joint pain|arthritis|gout": {"category": "pain", "subcategory": "joint"},
        "back pain|herniated disc|spinal stenosis": {"category": "pain", "subcategory": "back"},
        "neck pain|cervical|meningitis": {"category": "pain", "subcategory": "neck"},
        "c穩定|bronchitis": {"category": "respiratory", "subcategory": "cough"},
        "shortness of breath|dyspnea": {"category": "respiratory", "subcategory": "shortness of breath"},
        "wheezing|asthma": {"category": "respiratory", "subcategory": "wheezing"},
        "cystic fibrosis|bronchopulmonary dysplasia": {"category": "respiratory", "subcategory": "chronic lung"},
        "palpitations|arrhythmia": {"category": "cardiovascular", "subcategory": "palpitations"},
        "syncope|fainting": {"category": "cardiovascular", "subcategory": "syncope"},
        "edema|swelling|heart failure": {"category": "cardiovascular", "subcategory": "edema"},
        "atrial septal defect|ventricular septal defect|congenital heart": {"category": "cardiac", "subcategory": "congenital defect"},
        "nausea|gastritis|vomiting": {"category": "gastrointestinal", "subcategory": "nausea"},
        "diarrhea|ibs": {"category": "gastrointestinal", "subcategory": "diarrhea"},
        "dysphagia|esophageal": {"category": "gastrointestinal", "subcategory": "dysphagia"},
        "constipation|bowel": {"category": "gastrointestinal", "subcategory": "constipation"},
        "hirschsprung|volvulus|adhesions|bowel obstruction": {"category": "gastrointestinal", "subcategory": "bowel obstruction"},
        "weakness|neuropathy|myasthenia": {"category": "neurological", "subcategory": "weakness"},
        "confusion|delirium|dementia": {"category": "neurological", "subcategory": "confusion"},
        "numbness|tingling": {"category": "neurological", "subcategory": "numbness"},
        "tremors|parkinson": {"category": "neurological", "subcategory": "tremors"},
        "vertigo|dizziness|meniere": {"category": "neurological", "subcategory": "vertigo"},
        "epilepsy|seizure": {"category": "neurological", "subcategory": "seizures"},
        "rash|eczema|psoriasis|dermatitis": {"category": "skin", "subcategory": "rash"},
        "itching|pruritus|urticaria": {"category": "skin", "subcategory": "itching"},
        "facial swelling|angioedema": {"category": "skin", "subcategory": "facial swelling"},
        "fever|febrile|infection": {"category": "general", "subcategory": "fever"},
        "weight loss|cachexia": {"category": "general", "subcategory": "weight loss"},
        "fatigue|tiredness|chronic fatigue": {"category": "general", "subcategory": "fatigue"},
        "adhd|inattention|hyperactivity": {"category": "general", "subcategory": "inattention"},
        "vision changes|blindness|retinopathy|glaucoma": {"category": "sensory", "subcategory": "vision changes"},
        "sore throat|pharyngitis|tonsillitis": {"category": "infectious", "subcategory": "sore throat"},
        "edema|nephrotic|kidney disease": {"category": "renal", "subcategory": "edema"},
        "flank pain|kidney stone|pyelonephritis": {"category": "renal", "subcategory": "flank pain"},
        "depression|low mood|dysthymia": {"category": "psychiatric", "subcategory": "depression"},
        "anxiety|panic": {"category": "psychiatric", "subcategory": "anxiety"},
        "bleeding|hemophilia|thrombocytopenia": {"category": "hematologic", "subcategory": "bleeding"},
        "diabetes|hyperglycemia": {"category": "endocrinologic", "subcategory": "diabetes"},
        "hypothyroidism|hyperthyroidism|thyroid": {"category": "endocrinologic", "subcategory": "thyroid disorder"}
    }
    
    for pattern, mapping in symptom_mappings.items():
        if re.search(pattern, symptom.lower()):
            return mapping["category"], mapping["subcategory"]
    
    logging.warning(f"Unmapped symptom: {symptom}")
    return None, None

def scrape_pathways(source):
    """Scrape clinical pathways or guidelines from a given source."""
    pathways = {}
    base_url = source["url"]
    source_name = source["name"]
    
    if source.get("api"):
        # Handle PubMed API
        for symptom in [
            "headache", "chest pain", "abdominal pain", "cough", "shortness of breath",
            "palpitations", "nausea", "weakness", "rash", "edema", "fever", "diabetes", "seizure"
        ]:  # Limited set for testing
            category, subcategory = categorize_symptom(symptom)
            if not category or not subcategory:
                continue
            
            logging.info(f"Fetching PubMed articles for {symptom}")
            articles = fetch_pubmed_articles(symptom)
            if articles:
                pathway_data = extract_pubmed_content(articles, source_name)
                if pathway_data:
                    pathway_data["source"] = source_name
                    if category not in pathways:
                        pathways[category] = {}
                    pathways[category][subcategory] = pathway_data
        return pathways
    
    # Try primary URL (static first, then Selenium)
    html = fetch_page(base_url) or fetch_page(base_url, use_selenium=True)
    
    # Try fallback URLs
    if not html:
        for fallback in source["fallbacks"]:
            logging.info(f"Trying fallback URL for {source_name}: {fallback}")
            html = fetch_page(fallback) or fetch_page(fallback, use_selenium=True)
            if html:
                break
        if not html:
            logging.error(f"All URLs failed for {source_name}. Skipping.")
            return pathways
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Find pathway/guideline links (strict filtering)
    link_pattern = re.compile(r"(pain|fever|bleeding|cough|shortness of breath|palpitations|nausea|weakness|rash|edema|headache|chest|abdominal|joint|seizure|diarrhea|constipation|vertigo|diabetes|thyroid|asthma|pneumonia|stroke|condition|disease|guidance|management|treatment|symptom)", re.IGNORECASE)
    exclude_pattern = re.compile(r"(conditions we treat|view all|resource types|health and social care|oral and dental|sleep conditions|home|about|contact|search|privacy|terms|site map|schedule|appointment|mychart|careers|research|giving|call nationwide)", re.IGNORECASE)
    
    # Handle specific sources
    pathway_links = []
    if source_name == "Mayo Clinic":
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            alpha_url = f"https://www.mayoclinic.org/diseases-conditions/index?letter={letter}"
            alpha_html = fetch_page(alpha_url, use_selenium=True)
            if alpha_html:
                alpha_soup = BeautifulSoup(alpha_html, "html.parser")
                links = alpha_soup.find_all("a", href=link_pattern)
                pathway_links.extend([link for link in links if not exclude_pattern.search(link.get("href", "") or link.get_text(strip=True))])
    elif source_name == "CDC":
        az_url = "https://www.cdc.gov/az/"
        az_html = fetch_page(az_url, use_selenium=True)
        if az_html:
            az_soup = BeautifulSoup(az_html, "html.parser")
            links = az_soup.find_all("a", href=link_pattern)
            pathway_links.extend([link for link in links if not exclude_pattern.search(link.get("href", "") or link.get_text(strip=True))])
    elif source_name == "UpToDate":
        # UpToDate requires search-based crawling
        for term in ["headache", "chest pain", "fever"]:
            search_url = f"https://www.uptodate.com/contents/search?search={term}"
            search_html = fetch_page(search_url, use_selenium=True)
            if search_html:
                search_soup = BeautifulSoup(search_html, "html.parser")
                links = search_soup.find_all("a", href=link_pattern)
                pathway_links.extend([link for link in links if not exclude_pattern.search(link.get("href", "") or link.get_text(strip=True))])
    elif source_name == "ClinicalKey":
        # ClinicalKey uses browse conditions
        browse_html = fetch_page(base_url, use_selenium=True)
        if browse_html:
            browse_soup = BeautifulSoup(browse_html, "html.parser")
            links = browse_soup.find_all("a", href=link_pattern)
            pathway_links.extend([link for link in links if not exclude_pattern.search(link.get("href", "") or link.get_text(strip=True))])
    else:
        pathway_links = [link for link in soup.find_all("a", href=link_pattern) if not exclude_pattern.search(link.get("href", "") or link.get_text(strip=True))]
    
    if not pathway_links:
        logging.warning(f"No relevant links found for {source_name}. Page structure may have changed.")
    
    for link in pathway_links:
        title = link.get_text(strip=True).lower()
        href = link.get("href")
        if not href:
            continue
        
        # Construct full URL
        full_url = urljoin(base_url, href)
        
        # Extract symptom/condition
        symptom = re.sub(r"(clinical pathway|management of|evaluation and|pathway|care of|guidance on|for|treatment of|symptoms and causes|diagnosis and treatment|we treat|disease -)", "", title, flags=re.IGNORECASE).strip()
        if not symptom or len(symptom) < 3:
            continue
        
        # Categorize symptom
        category, subcategory = categorize_symptom(symptom)
        if not category or not subcategory:
            continue
        
        logging.info(f"Processing {source_name} pathway: {symptom} ({full_url})")
        
        # Initialize pathway
        pathway_data = {
            "differentials": [],
            "workup": {"urgent": [], "routine": []},
            "management": {"symptomatic": [], "definitive": []},
            "source": source_name
        }
        
        # Handle PDF links
        if full_url.endswith(".pdf"):
            pdf_filename = f"temp_{source_name}_{symptom.replace(' ', '_')}.pdf"
            if download_pdf(full_url, pdf_filename):
                extracted_data = extract_pdf_content(pdf_filename, source_name)
                if extracted_data:
                    pathway_data.update(extracted_data)
                os.remove(pdf_filename)
        
        # Handle HTML content
        html_content = fetch_page(full_url) or fetch_page(full_url, use_selenium=True)
        if html_content:
            sub_soup = BeautifulSoup(html_content, "html.parser")
            content = sub_soup.get_text(strip=True).lower()
            
            # Source-specific extraction
            diff_pattern = {
                "Mayo Clinic": r"(?:causes include|possible causes|consider|may include)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "CDC": r"(?:caused by|risk factors|may include)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "WHO": r"(?:causes|etiologies|may include)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "UpToDate": r"(?:differential diagnosis|etiologies|may include)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "ClinicalKey": r"(?:causes|differential|consider)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "default": r"(?:differential diagnosis|possible causes|consider|may include|causes include)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
            }.get(source_name, "default")
            urgent_pattern = {
                "Mayo Clinic": r"(?:seek immediate|emergency care|urgent|prompt)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "CDC": r"(?:seek care immediately|urgent|what to do if)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "WHO": r"(?:urgent|immediate action|emergency)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "UpToDate": r"(?:urgent|immediate|prompt)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "ClinicalKey": r"(?:emergency|urgent care|prompt)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "default": r"(?:urgent|immediate|initial evaluation|emergency|prompt)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
            }.get(source_name, "default")
            routine_pattern = {
                "Mayo Clinic": r"(?:diagnosis includes|tests include|diagnostic|follow-up)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "CDC": r"(?:diagnosed by|tests for|routine)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "WHO": r"(?:diagnostic|investigation|assessment)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "UpToDate": r"(?:diagnostic evaluation|tests|assessment)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "ClinicalKey": r"(?:diagnostic|tests|workup)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "default": r"(?:routine|diagnostic|follow-up|investigation|assessment|tests include)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
            }.get(source_name, "default")
            symp_pattern = {
                "Mayo Clinic": r"(?:relieve symptoms|symptom relief|initial treatment)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "CDC": r"(?:manage symptoms|symptom management|relief)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "WHO": r"(?:symptomatic treatment|supportive care)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "UpToDate": r"(?:symptom management|supportive treatment)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "ClinicalKey": r"(?:symptomatic care|relief|initial treatment)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "default": r"(?:symptomatic|supportive|initial treatment|relief|symptom management)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
            }.get(source_name, "default")
            def_pattern = {
                "Mayo Clinic": r"(?:treatment includes|long-term care|specific treatment)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "CDC": r"(?:treatment for|cure|management includes)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "WHO": r"(?:guideline|definitive treatment|long-term management)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "UpToDate": r"(?:treatment guidelines|definitive care|management)[:CERN]*([\w\s,;-]+?)(?:\n|\.|$)",
                "ClinicalKey": r"(?:definitive treatment|management|therapy)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)",
                "default": r"(?:definitive|specific|long-term|curative|guideline|treatment includes)[:\s]*([\w\s,;-]+?)(?:\n|\.|$)"
            }.get(source_name, "default")
            
            # Extract differentials
            diff_matches = re.findall(diff_pattern, content, re.IGNORECASE)
            pathway_data["differentials"] = [m.strip() for match in diff_matches for m in match.split(",") if len(m.strip()) > 3 and "condition" not in m.lower()][:4]
            
            # Extract workup
            urgent_matches = re.findall(urgent_pattern, content, re.IGNORECASE)
            routine_matches = re.findall(routine_pattern, content, re.IGNORECASE)
            pathway_data["workup"]["urgent"] = [u.strip() for match in urgent_matches for u in match.split(",") if u.strip() and len(u.strip()) > 5 and "call" not in u.lower()][:3]
            pathway_data["workup"]["routine"] = [r.strip() for match in routine_matches for r in match.split(",") if r.strip() and len(r.strip()) > 5 and "learn" not in r.lower()][:4]
            
            # Extract management
            symp_matches = re.findall(symp_pattern, content, re.IGNORECASE)
            def_matches = re.findall(def_pattern, content, re.IGNORECASE)
            pathway_data["management"]["symptomatic"] = [s.strip() for match in symp_matches for s in match.split(",") if s.strip() and len(s.strip()) > 5 and "lifestyle" not in s.lower()][:3]
            pathway_data["management"]["definitive"] = [d.strip() for match in def_matches for d in match.split(",") if d.strip() and len(d.strip()) > 5 and "research" not in d.lower()][:3]
        
        # Add non-empty pathways
        if pathway_data["differentials"] or any(pathway_data["workup"].values()) or any(pathway_data["management"].values()):
            if category not in pathways:
                pathways[category] = {}
            pathways[category][subcategory] = pathway_data
    
    return pathways

def merge_pathways(new_pathways):
    """Merge scraped pathways with existing clinical_pathways.json, preserving better data."""
    try:
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r") as f:
                existing = json.load(f)
        else:
            existing = {}
        
        # Merge new pathways, keeping richer data
        for category, subcategories in new_pathways.items():
            if category not in existing:
                existing[category] = {}
            for subcategory, new_data in subcategories.items():
                # Skip if new data is sparse
                if not new_data["differentials"] and not any(new_data["workup"].values()) and not any(new_data["management"].values()):
                    continue
                # Keep existing if it has more content
                existing_data = existing[category].get(subcategory, {})
                existing_score = len(existing_data.get("differentials", [])) + sum(len(v) for v in existing_data.get("workup", {}).values()) + sum(len(v) for v in existing_data.get("management", {}).values())
                new_score = len(new_data["differentials"]) + sum(len(v) for v in new_data["workup"].values()) + sum(len(v) for v in new_data["management"].values())
                if existing_score > new_score and existing_data:
                    logging.info(f"Keeping existing data for {category}/{subcategory} (score {existing_score} > {new_score})")
                    continue
                existing[category][subcategory] = new_data
        
        # Write back to file
        with open(OUTPUT_FILE, "w") as f:
            json.dump(existing, f, indent=2)
        logging.info(f"Updated {OUTPUT_FILE} with {len(new_pathways)} new categories")
    except Exception as e:
        logging.error(f"Failed to merge pathways: {e}")

def main():
    """Main function to run the scraper."""
    logging.info("Starting clinical pathways scraper...")
    all_pathways = {}
    
    for source in SOURCES:
        logging.info(f"Scraping {source['name']}...")
        pathways = scrape_pathways(source)
        for category, subcategories in pathways.items():
            if category not in all_pathways:
                all_pathways[category] = {}
            all_pathways[category].update(subcategories)
    
    if all_pathways:
        merge_pathways(all_pathways)
    else:
        logging.warning("No pathways scraped from any source.")
    logging.info("Scraping complete.")

if __name__ == "__main__":
    main()