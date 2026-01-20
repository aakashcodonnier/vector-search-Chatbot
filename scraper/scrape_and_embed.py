import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from database.db import get_connection
import time
import logging
from datetime import datetime
from urllib.parse import urljoin

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ================= CONFIG =================
BASE_URL = "https://phoreveryoung.wordpress.com/category/case-studies/"
HEADERS = {"User-Agent": "Mozilla/5.0 (CaseStudyBot/1.0)"}
MIN_CONTENT_LENGTH = 200
PAGE_DELAY = 1
ARTICLE_DELAY = 2

model = SentenceTransformer("all-MiniLM-L6-v2")

# ================= DB =================
conn = get_connection()
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS articles (
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

# ================= SCRAPER =================
start_time = time.time()
total_articles = 0
page_url = BASE_URL

logger.info("=" * 60)
logger.info("🚀 SCRAPER STARTED")
logger.info("=" * 60)

while page_url:
    logger.info(f"\n📄 Scraping page: {page_url}")

    try:
        response = requests.get(page_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"❌ Failed to fetch page: {e}")
        break

    soup = BeautifulSoup(response.text, "html.parser")

    # ✅ WordPress Twenty Fourteen Theme Specific Selectors
    articles = (soup.select("article.post.hentry") or 
                soup.select("article.post") or 
                soup.select("article.type-post") or 
                soup.select(".post") or 
                soup.find_all("article"))
    
    # Filter for case study related content
    articles = [art for art in articles 
               if any(keyword in art.get_text().lower() 
                     for keyword in ['case', 'study', 'research', 'patient', 'treatment', 'cancer'])]

    if not articles:
        logger.info("✅ No articles found. Scraping finished.")
        break

    logger.info(f"Found {len(articles)} articles")

    for art in articles:
        try:
            # WordPress Twenty Fourteen Theme Title Selectors
            title_tag = (art.find("h1", class_="entry-title") or 
                        art.find("h2", class_="entry-title") or 
                        art.select_one(".entry-title a") or
                        art.find("h1") or 
                        art.find("h2"))
            
            if not title_tag:
                continue
            
            # Look for link in title or as direct child
            title_link = title_tag.find("a")
            if not title_link:
                # Try getting href from title if it's an anchor
                if title_tag.name == 'a':
                    title_link = title_tag
                else:
                    # Try finding first link in title div
                    title_wrapper = art.select_one(".entry-header") or art
                    title_link = title_wrapper.find("a")
            
            if not title_link:
                continue

            url = title_link["href"]

            # ⚡ FAST duplicate check (URL unique index)
            cur.execute("SELECT 1 FROM articles WHERE url=%s LIMIT 1", (url,))
            if cur.fetchone():
                logger.info(f"⚠️ Duplicate skipped: {url}")
                continue

            # ===== Detail page =====
            detail_res = requests.get(url, headers=HEADERS, timeout=15)
            detail_res.raise_for_status()
            detail = BeautifulSoup(detail_res.text, "html.parser")

            # WordPress Twenty Fourteen Theme Title Selectors
            title_elem = (detail.find("h1", class_="entry-title") or 
                         detail.find("h1") or 
                         detail.find("h2", class_="entry-title") or 
                         detail.find("h2") or
                         detail.select_one(".entry-header h1") or
                         detail.select_one(".entry-header h2"))
            title = title_elem.get_text(strip=True) if title_elem else ""

            date = detail.find("time", class_="entry-date")
            date = date.get_text(strip=True) if date else ""

            author = detail.find("span", class_="author")
            author = author.get_text(strip=True) if author else ""

            # WordPress Twenty Fourteen Theme Category Selectors
            category_links = (detail.select(".cat-links a") or 
                             detail.select(".categories a") or 
                             detail.select(".tags a") or 
                             detail.select(".category a") or
                             detail.select("footer.entry-meta a") or
                             detail.select("span.tag-links a"))
            categories = ", ".join(c.get_text(strip=True) for c in category_links) if category_links else "uncategorized"

            # WordPress Twenty Fourteen Theme Content Selectors
            content_div = (detail.find("div", class_="entry-content") or 
                          detail.find("div", class_="post-content") or 
                          detail.find("div", class_="content") or 
                          detail.find("div", class_="site-content") or 
                          detail.find("article") or 
                          detail.find(".post-body") or
                          detail.select_one(".entry-content"))
            
            if not content_div:
                continue

            # Extract content from multiple element types (not just <p>)
            content_elements = content_div.find_all(["p", "div", "span", "section", "article"])
            content_parts = []
            
            for elem in content_elements:
                text = elem.get_text(" ", strip=True)
                # Skip very short or irrelevant content
                if len(text) > 20 and not any(skip_word in text.lower() 
                                           for skip_word in ['advertisement', 'sidebar', 'menu', 'footer']):
                    content_parts.append(text)
            
            content = " ".join(content_parts)

            if len(content) < MIN_CONTENT_LENGTH:
                logger.info(f"⚠️ Skipped short article ({len(content)} chars)")
                continue

            # ===== Embedding =====
            embedding = model.encode(content).tolist()

            cur.execute("""
                INSERT INTO articles
                (title, url, date, author, categories, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                title,
                url,
                date,
                author,
                categories,
                content,
                str(embedding)
            ))

            conn.commit()
            total_articles += 1

            logger.info(f"✅ INSERTED: {title}")
            logger.info(f"   Content: {len(content)} chars | Embedding: {len(embedding)} dims")

            time.sleep(ARTICLE_DELAY)

        except Exception as e:
            logger.error(f"❌ Article error: {e}")
            continue

    # ✅ NEXT PAGE DETECTION (WordPress correct way)
    next_link = soup.find("a", class_="next")
    page_url = urljoin(BASE_URL, next_link["href"]) if next_link else None

    time.sleep(PAGE_DELAY)

# ================= END =================
conn.close()
total_time = time.time() - start_time

logger.info("=" * 60)
logger.info("🎉 SCRAPER COMPLETED")
logger.info(f"Total articles scraped: {total_articles}")
logger.info(f"Total time: {total_time/60:.2f} minutes")
logger.info("=" * 60)
