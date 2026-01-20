import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from database.db import get_connection
import time
import logging
from urllib.parse import urljoin
import warnings

warnings.filterwarnings("ignore")

# ================= CONTENT EXTRACTOR =================
def extract_clean_article_content(soup):
    selectors = [
        "article",
        "div.entry-content",
        "div.elementor-widget-theme-post-content",
        "main"
    ]

    content_root = None
    for sel in selectors:
        content_root = soup.select_one(sel)
        if content_root:
            break

    if not content_root:
        return ""

    elements = content_root.find_all(
        ["p", "h1", "h2", "h3", "h4", "li"],
        recursive=True
    )

    content_parts = []
    for el in elements:
        text = el.get_text(" ", strip=True)

        if len(text) < 30:
            continue

        if any(skip in text.lower() for skip in [
            "share this",
            "related",
            "author",
            "posted on",
            "subscribe",
            "navigation",
            "footer",
            "copyright",
            "all rights reserved",
            "privacy policy",
            "terms of service",
            "cookie",
            "menu",
            "search",
            "leave a comment",
            "reply",
            "previous post",
            "next post",
            "facebook",
            "twitter",
            "linkedin",
            "email"
        ]):
            continue

        content_parts.append(text)

    return "\n".join(content_parts)


# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ================= CONFIG =================
CASE_STUDY_BASE_URL = "https://phoreveryoung.wordpress.com/category/case-studies/"
HEADERS = {"User-Agent": "Mozilla/5.0 (MultiSiteBot/1.0)"}
MIN_CONTENT_LENGTH = 300
PAGE_DELAY = 1
ARTICLE_DELAY = 2

model = SentenceTransformer("all-MiniLM-L6-v2")

# ================= DB =================
conn = get_connection()
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS dr_young_all_articles (
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    title TEXT,
    url TEXT UNIQUE,
    date VARCHAR(100),
    author VARCHAR(255),
    categories TEXT,
    content LONGTEXT,
    embedding LONGTEXT
)
""")
conn.commit()

# ================= CASE STUDY SCRAPER =================
def scrape_case_studies():
    total_articles = 0
    page_url = CASE_STUDY_BASE_URL

    logger.info("🚀 CASE STUDY SCRAPER STARTED")

    while page_url:
        response = requests.get(page_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        articles = soup.find_all("article")
        if not articles:
            break

        for art in articles:
            title_link = art.find("a", href=True)
            if not title_link:
                continue

            url = title_link["href"]

            cur.execute("SELECT 1 FROM dr_young_all_articles WHERE url=%s", (url,))
            if cur.fetchone():
                continue

            try:
                detail_res = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=10,
                    verify=False
                )
            except Exception:
                continue

            detail_soup = BeautifulSoup(detail_res.text, "html.parser")

            title_elem = detail_soup.find("h1")
            title = title_elem.get_text(strip=True) if title_elem else ""

            content = extract_clean_article_content(detail_soup)
            if len(content) < MIN_CONTENT_LENGTH:
                continue

            embedding = model.encode(content).tolist()

            cur.execute("""
                INSERT INTO dr_young_all_articles
                (title, url, content, embedding)
                VALUES (%s, %s, %s, %s)
            """, (title, url, content, str(embedding)))

            conn.commit()
            total_articles += 1
            logger.info(f"✅ INSERTED CASE STUDY: {title}")

            time.sleep(ARTICLE_DELAY)

        next_link = soup.find("a", class_="next")
        page_url = urljoin(CASE_STUDY_BASE_URL, next_link["href"]) if next_link else None
        time.sleep(PAGE_DELAY)

    return total_articles


# ================= DR ROBERT YOUNG SCRAPER =================
def scrape_dr_young_all_categories():
    categories = [
        "blog",
        "articles",
        "clean-eating",
        "digestive-health",
        "womens-health",
        "corona-virus"
    ]

    total_articles = 0
    logger.info("🚀 DR ROBERT YOUNG SCRAPER STARTED")

    for category in categories:
        category_url = f"https://drrobertyoung.com/{category}/"

        try:
            response = requests.get(
                category_url,
                headers=HEADERS,
                timeout=10,
                verify=False
            )
        except Exception:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        links = soup.find_all("a", href=True)
        post_urls = set()

        for a in links:
            href = a["href"]

            if not href.startswith("https://drrobertyoung.com/"):
                continue

            if any(skip in href for skip in [
                "/wp-content/",
                "/category/",
                "/tag/",
                "/page/",
                "#",
                "?",
                "/feed",
                "/comment",
                ".jpg",
                ".png"
            ]):
                continue

            if href.count("-") < 3:
                continue

            post_urls.add(href)

        for url in post_urls:
            cur.execute("SELECT 1 FROM dr_young_all_articles WHERE url=%s", (url,))
            if cur.fetchone():
                continue

            try:
                res = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=10,
                    verify=False
                )
            except Exception:
                logger.warning(f"⚠️ Skipped URL: {url}")
                continue

            soup = BeautifulSoup(res.text, "html.parser")

            title_elem = soup.find("h1")
            title = title_elem.get_text(strip=True) if title_elem else url.split("/")[-1]

            content = extract_clean_article_content(soup)
            if len(content) < MIN_CONTENT_LENGTH:
                continue

            embedding = model.encode(content).tolist()

            cur.execute("""
                INSERT INTO dr_young_all_articles
                (title, url, content, embedding)
                VALUES (%s, %s, %s, %s)
            """, (title, url, content, str(embedding)))

            conn.commit()
            total_articles += 1
            logger.info(f"✅ INSERTED ARTICLE: {title}")

            time.sleep(ARTICLE_DELAY)

    return total_articles


# ================= MAIN =================
try:
    cs = scrape_case_studies()
    time.sleep(3)
    dr = scrape_dr_young_all_categories()

    logger.info("🎉 SCRAPING COMPLETED")
    logger.info(f"Case Studies: {cs}")
    logger.info(f"Dr Young Articles: {dr}")
    logger.info(f"TOTAL: {cs + dr}")

finally:
    conn.close()
