#!/usr/bin/env python3
"""
Test script to scrape symptoms for conditions from Mayo Clinic and UpToDateOnline
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json
from pathlib import Path
import random
import time
try:
    from fuzzywuzzy import fuzz
except ImportError:
    logging.warning("fuzzywuzzy not installed. Install with `pip install fuzzywuzzy`")
    fuzz = None
try:
    from playwright.async_api import async_playwright
except ImportError:
    logging.warning("playwright not installed. Install with `pip install playwright` and run `playwright install`")
    async_playwright = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'request_timeout': 40,  # Initial timeout
    'scrape_delay': 30,    # Increased to avoid rate-limiting
    'conditions_file': Path('conditions.json'),
    'output_file': Path('symptoms.json'),
    'max_retries': 8,
}

# User agents for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
]

# URL mappings for Mayo Clinic
MAYO_URL_MAPPINGS = {
    'heart failure': [('heart-failure', '20373187'), ('heart-failure', '20373142')],
    'malaria': [('malaria', '20351184')],
    'gout': [('gout', '20372897')],
    'diabetes mellitus type 2': [('type-2-diabetes', '20351197'), ('type-2-diabetes', '20351193')],
    'pneumonia': [('pneumonia', '20351697'), ('pneumonia', '20351704')],
    'migraine': [('migraine-headache', '20360201')],
    'urinary tract infection': [('urinary-tract-infection', '20353447')],
    'chronic obstructive pulmonary disease': [('copd', '20353679')],
    'irritable bowel syndrome': [('irritable-bowel-syndrome', '20360016')],
    'anemia': [('anemia', '20351360')],
    'depression': [('depression', '20356007')],
    'stroke': [('stroke', '20350113')],
    'kidney stones': [('kidney-stones', '20355755')]
}

# Search terms for UpToDateOnline
UPTODATE_SEARCH_TERMS = {
    'heart failure': ['heart failure', 'نارسایی قلبی', 'نارسایی قلب'],
    'malaria': ['malaria', 'مالاریا'],
    'gout': ['gout', 'نقرس'],
    'diabetes mellitus type 2': ['type 2 diabetes', 'دیابت نوع ۲', 'دیابت نوع دوم'],
    'pneumonia': ['pneumonia', 'پنومونی', 'سینه پهلو'],
    'migraine': ['migraine', 'میگرن', 'سردرد میگرنی'],
    'urinary tract infection': ['urinary tract infection', 'عفونت ادراری', 'عفونت مجاری ادراری'],
    'chronic obstructive pulmonary disease': ['copd', 'بیماری انسدادی مزمن ریه'],
    'irritable bowel syndrome': ['irritable bowel syndrome', 'سندرم روده تحریک‌پذیر', 'روده تحریک‌پذیر'],
    'anemia': ['anemia', 'کم‌خونی', 'آنمی'],
    'depression': ['depression', 'افسردگی'],
    'stroke': ['stroke', 'سکته مغزی', 'سکته'],
    'kidney stones': ['kidney stones', 'سنگ کلیه', 'سنگ‌های کلیوی']
}

# Persian-to-English symptom translation
SYMPTOM_TRANSLATIONS = {
    'تب': 'fever',
    'لرز': 'chills',
    'درد': 'pain',
    'سردرد': 'headache',
    'تهوع': 'nausea',
    'استفراغ': 'vomiting',
    'خستگی': 'fatigue',
    'تنگی نفس': 'shortness of breath',
    'سرگیجه': 'dizziness',
    'درد مفاصل': 'joint pain',
    'درد شکمی': 'abdominal pain',
    'اسهال': 'diarrhea',
    'تورم': 'swelling',
    'ضعف': 'weakness'
}

def merge_similar_symptoms(symptoms: list[str], threshold: int = 85) -> list[str]:
    """Merge similar symptoms using fuzzy matching"""
    if not fuzz:
        logger.warning("fuzzywuzzy not available, skipping symptom merging")
        return symptoms
    merged = []
    used = set()
    for i, s1 in enumerate(symptoms):
        if i in used:
            continue
        merged.append(s1)
        used.add(i)
        for j, s2 in enumerate(symptoms[i+1:], i+1):
            if j not in used and fuzz.ratio(s1, s2) > threshold:
                used.add(j)
    return merged

def translate_symptom(symptom: str) -> str:
    """Translate Persian symptoms to English if possible"""
    return SYMPTOM_TRANSLATIONS.get(symptom.strip(), symptom.strip())

@retry(
    stop=stop_after_attempt(CONFIG['max_retries']),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    retry=retry_if_exception_type((aiohttp.ClientError, aiohttp.ServerTimeoutError, aiohttp.ClientConnectorDNSError, asyncio.TimeoutError))
)
async def scrape_mayo_clinic_symptoms(condition: str, session: aiohttp.ClientSession, attempt: int = 1) -> list[str]:
    """Scrape symptoms from Mayo Clinic with dynamic timeout"""
    try:
        timeout = CONFIG['request_timeout'] + (attempt * 10)  # Increase timeout per attempt
        mappings = MAYO_URL_MAPPINGS.get(condition.lower(), [(condition.lower().replace(' ', '-'), '20355755')])
        symptoms = []
        for mayo_key, syc_id in mappings:
            url = f"https://www.mayoclinic.org/diseases-conditions/{mayo_key}/symptoms-causes/syc-{syc_id}"
            logger.info(f"Scraping symptoms for {condition} from {url} (attempt {attempt}, timeout {timeout}s)")

            headers = {
                'User-Agent': random.choice(USER_AGENTS),
                'Accept': 'text/html',
                'Referer': 'https://www.mayoclinic.org/'
            }
            async with session.get(url, timeout=timeout, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: Status {response.status}")
                    continue
                final_url = str(response.url)
                if mayo_key not in final_url:
                    logger.warning(f"Redirected to wrong condition {final_url}, skipping")
                    continue
                html = await response.text()
                if final_url != url:
                    logger.info(f"Redirected to {final_url}")

                soup = BeautifulSoup(html, 'html.parser')
                symptoms_section = soup.find('h2', string=re.compile('^Symptoms$', re.I))
                if symptoms_section:
                    for ul in symptoms_section.find_all_next('ul'):
                        if ul.find_previous('h2') != symptoms_section:
                            break
                        for li in ul.find_all('li'):
                            symptom = li.get_text(strip=True).lower()
                            if symptom:
                                symptom = symptom.split('.')[0] if '.' in symptom else symptom
                                symptoms.append(symptom)
                    if not symptoms:
                        logger.warning(f"No <ul> found under Symptoms section for {condition}")
                    else:
                        break

        symptoms = list(dict.fromkeys(
            s for s in symptoms
            if len(s) > 5 and all(phrase not in s for phrase in [
                'see a doctor', 'symptoms', 'ureter', 'form in', 'at that point', 'if you have',
                'call', 'contact', 'suicide', 'hotline', 'mayo clinic', 'crisis line',
                'ask the person', 'check with', 'emergency', 'reach out',
                'is there a link', 'understanding the issues', 'what does it mean',
                'after a head injury', 'after age 50', 'could be a sign of'
            ])
        ))

        symptoms = merge_similar_symptoms(symptoms)
        logger.info(f"Scraped symptoms for {condition} from Mayo Clinic: {symptoms}")
        return symptoms

    except Exception as e:
        logger.error(f"Error scraping for {condition} from Mayo Clinic: {str(e)}", exc_info=True)
        raise

async def scrape_with_playwright(url: str) -> str:
    """Scrape URL using Playwright for JavaScript-rendered content"""
    if not async_playwright:
        logger.warning("Playwright not available, skipping browser-based scraping")
        return ""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, timeout=60000)
            html = await page.content()
            await browser.close()
            return html
    except Exception as e:
        logger.error(f"Error scraping {url} with Playwright: {str(e)}")
        return ""

@retry(
    stop=stop_after_attempt(CONFIG['max_retries']),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    retry=retry_if_exception_type((aiohttp.ClientError, aiohttp.ServerTimeoutError, aiohttp.ClientConnectorDNSError))
)
async def scrape_uptodate_symptoms(condition: str, session: aiohttp.ClientSession, use_playwright: bool = True) -> list[str]:
    """Scrape symptoms from UpToDateOnline"""
    try:
        search_terms = UPTODATE_SEARCH_TERMS.get(condition.lower(), [condition])
        symptoms = []
        for term in search_terms:
            search_url = f"https://www.uptodateonline.ir/contents/search.htm?search={term.replace(' ', '+')}"
            logger.info(f"Scraping symptoms for {condition} from {search_url}")

            headers = {
                'User-Agent': random.choice(USER_AGENTS),
                'Accept': 'text/html',
                'Referer': 'https://www.uptodateonline.ir/'
            }
            html = ""
            if use_playwright and async_playwright:
                html = await scrape_with_playwright(search_url)
            else:
                async with session.get(search_url, timeout=CONFIG['request_timeout'], headers=headers) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch {search_url}: Status {response.status}")
                        continue
                    html = await response.text()

            if not html:
                logger.warning(f"No HTML retrieved for {condition} with term {term}")
                continue

            soup = BeautifulSoup(html, 'html.parser')
            search_snippet = str(soup.find('body'))[:500]
            logger.debug(f"Search page snippet for {condition} (term: {term}): {search_snippet}")

            article_link = None
            for a in soup.find_all('a', href=True):
                link_text = a.get_text().lower()
                link_href = a['href'].lower()
                if (any(t.lower() in link_text or t.lower() in link_href for t in [term, condition]) and
                    a['href'].startswith('https://www.uptodateonline.ir/contents/')):
                    article_link = a['href']
                    break

            if not article_link:
                logger.warning(f"No relevant article link found for {condition} with term {term}")
                continue

            logger.info(f"Following article link for {condition}: {article_link}")
            article_html = ""
            if use_playwright and async_playwright:
                article_html = await scrape_with_playwright(article_link)
            else:
                async with session.get(article_link, timeout=CONFIG['request_timeout'], headers=headers) as article_response:
                    if article_response.status != 200:
                        logger.warning(f"Failed to fetch {article_link}: Status {article_response.status}")
                        continue
                    article_html = await article_response.text()

            soup = BeautifulSoup(article_html, 'html.parser')
            article_snippet = str(soup.find('body'))[:500]
            logger.debug(f"Article page snippet for {condition}: {article_snippet}")

            content_div = soup.find('div', class_=re.compile('content|entry-content|post-content|article|main|topic-content'))
            if content_div:
                symptom_section = soup.find(['h2', 'h3'], string=re.compile('علائم|نشانه|Symptoms|علامت|Clinical manifestations', re.I))
                if symptom_section:
                    for ul in symptom_section.find_all_next('ul'):
                        if ul.find_previous(['h2', 'h3']) != symptom_section:
                            break
                        for li in ul.find_all('li'):
                            symptom = li.get_text(strip=True).lower()
                            if symptom:
                                symptom = symptom.split('.')[0] if '.' in symptom else symptom
                                symptoms.append(translate_symptom(symptom))
                else:
                    for p in content_div.find_all('p'):
                        text = p.get_text(strip=True).lower()
                        if any(keyword in text for keyword in ['علائم', 'نشانه', 'symptoms', 'علامت', 'clinical manifestations']):
                            symptoms.extend([translate_symptom(s.strip()) for s in text.split('،') if s.strip()])
                    for li in content_div.find_all('li'):
                        symptom = li.get_text(strip=True).lower()
                        if symptom:
                            symptoms.append(translate_symptom(symptom))
            else:
                logger.warning(f"No content div found for {condition} at {article_link}")

            if symptoms:
                break

        symptoms = list(dict.fromkeys(
            s for s in symptoms
            if len(s) > 2 and all(phrase not in s for phrase in [
                'مشاوره', 'پزشک', 'تماس', 'اورژانس', 'بیمارستان', 'درمان', 'تشخیص',
                'see a doctor', 'call', 'contact', 'emergency', 'hospital', 'treatment', 'diagnosis'
            ])
        ))

        if not symptoms:
            logger.warning(f"No symptoms extracted for {condition} from UpToDateOnline")

        symptoms = merge_similar_symptoms(symptoms)
        logger.info(f"Scraped symptoms for {condition} from UpToDateOnline: {symptoms}")
        return symptoms

    except Exception as e:
        logger.error(f"Error scraping for {condition} from UpToDateOnline: {str(e)}", exc_info=True)
        return []

async def load_conditions() -> list[str]:
    """Load conditions from conditions.json"""
    try:
        with CONFIG['conditions_file'].open('r') as f:
            data = json.load(f)
        conditions = list(data.keys())
        logger.info(f"Loaded {len(conditions)} conditions: {conditions}")
        return conditions
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load conditions.json: {str(e)}")
        return list(MAYO_URL_MAPPINGS.keys())

async def save_symptoms(symptoms_dict: dict):
    """Save scraped symptoms to JSON file"""
    try:
        with CONFIG['output_file'].open('w', encoding='utf-8') as f:
            json.dump(symptoms_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved symptoms to {CONFIG['output_file']}")
    except Exception as e:
        logger.error(f"Failed to save symptoms to {CONFIG['output_file']}: {str(e)}")

async def test_scrape():
    """Scrape symptoms from Mayo Clinic and UpToDateOnline, combining results"""
    conditions = await load_conditions()
    symptoms_dict = {}
    async with aiohttp.ClientSession() as session:
        for condition in conditions:
            try:
                mayo_symptoms = await scrape_mayo_clinic_symptoms(condition, session)
                uptodate_symptoms = await scrape_uptodate_symptoms(condition, session, use_playwright=async_playwright is not None)
                combined_symptoms = list(dict.fromkeys(mayo_symptoms + uptodate_symptoms))
                combined_symptoms = merge_similar_symptoms(combined_symptoms)
                symptoms_dict[condition] = {
                    'symptoms': combined_symptoms,
                    'mayo_clinic': mayo_symptoms,
                    'uptodateonline': uptodate_symptoms,
                    'status': 'success' if combined_symptoms else 'failed',
                    'error': None
                }
                print(f"Symptoms for {condition}: {combined_symptoms}")
            except Exception as e:
                logger.error(f"Failed to scrape for {condition}: {str(e)}")
                symptoms_dict[condition] = {
                    'symptoms': [],
                    'mayo_clinic': [],
                    'uptodateonline': [],
                    'status': 'failed',
                    'error': str(e)
                }
            await asyncio.sleep(CONFIG['scrape_delay'] + random.uniform(0, 5))

    await save_symptoms(symptoms_dict)

if __name__ == "__main__":
    try:
        asyncio.run(test_scrape())
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}", exc_info=True)