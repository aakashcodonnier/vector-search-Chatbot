#!/usr/bin/env python3
"""
Web Scraper and Embedding Tool for Dr. Robert Young's Content

This module scrapes articles from multiple sources (case studies and Dr. Robert Young's blog)
and stores them in a database with vector embeddings for semantic search.

Features:
- Scrapes content from WordPress blog and case studies
- Extracts clean article content with noise filtering
- Generates semantic embeddings using sentence-transformers
- Stores structured data in MySQL database
- Handles rate limiting and respectful scraping practices
- Supports multiple content sources and categories

Scraping Sources:
1. Case Studies: https://phoreveryoung.wordpress.com/category/case-studies/
2. Dr. Robert Young Blog Categories: Multiple categories from drrobertyoung.com

Data Processing Pipeline:
1. Extract raw HTML content
2. Clean and filter relevant text
3. Generate vector embeddings
4. Store in database with metadata
"""

# Standard library imports
import time           # Timing and delays for respectful scraping
import logging        # Logging for monitoring and debugging
import warnings       # Warning suppression for cleaner output
from urllib.parse import urljoin  # URL joining utilities

# Third-party imports
import requests                    # HTTP requests for web scraping
from bs4 import BeautifulSoup     # HTML parsing and content extraction
from sentence_transformers import SentenceTransformer  # Vector embedding generation

# Local imports
from database.db import get_connection  # Database connection utility

# Suppress SSL warnings for cleaner output during scraping
warnings.filterwarnings("ignore")

# Configure logging for detailed progress tracking
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration constants for scraping behavior
CASE_STUDY_BASE_URL = "https://phoreveryoung.wordpress.com/category/case-studies/"
DR_YOUNG_BASE_URL = "https://drrobertyoung.com/"
HEADERS = {"User-Agent": "Mozilla/5.0 (MultiSiteBot/1.0)"}  # Identify bot properly
MIN_CONTENT_LENGTH = 300  # Minimum content length to store an article (characters)
PAGE_DELAY = 1            # Delay between pages (seconds) for rate limiting
ARTICLE_DELAY = 2         # Delay between articles (seconds) for respectful scraping

# Initialize sentence transformer model for embeddings
# Using all-MiniLM-L6-v2 for efficient sentence embeddings (22.7M parameters)
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_clean_article_content(soup):
    """
    Extract clean article content from a BeautifulSoup object
    
    This function attempts to find the main content of an article by trying various
    common HTML selectors and filtering out unwanted elements like navigation,
    advertisements, and footer content.
    
    Args:
        soup (BeautifulSoup): Parsed HTML document
        
    Returns:
        str: Cleaned article content or empty string if no content found
    """
    # Try common selectors for article content in order of preference
    selectors = [
        "article",                           # Standard article tag
        "div.entry-content",                 # WordPress entry content
        "div.elementor-widget-theme-post-content",  # Elementor theme content
        "main"                               # Main content area
    ]

    content_root = None
    for sel in selectors:
        content_root = soup.select_one(sel)
        if content_root:
            break

    # Return empty if no content container found
    if not content_root:
        return ""

    # Extract text from common content elements
    elements = content_root.find_all(
        ["p", "h1", "h2", "h3", "h4", "li"],  # Text-containing elements
        recursive=True
    )

    content_parts = []
    for el in elements:
        text = el.get_text(" ", strip=True)

        # Skip short texts that are likely navigation or formatting
        if len(text) < 30:
            continue

        # Skip unwanted content sections commonly found in footers/menus
        if any(skip in text.lower() for skip in [
            "share this", "related", "author", "posted on", "subscribe",
            "navigation", "footer", "copyright", "all rights reserved",
            "privacy policy", "terms of service", "cookie", "menu",
            "search", "leave a comment", "reply", "previous post",
            "next post", "facebook", "twitter", "linkedin", "email"
        ]):
            continue

        content_parts.append(text)

    return "\n".join(content_parts)

# Establish database connection and cursor
conn = get_connection()
cur = conn.cursor()

# Create articles table if it doesn't exist
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


def scrape_case_studies():
    """
    Scrape case studies from phoreveryoung.wordpress.com
    
    This function scrapes case study articles from the WordPress site,
    extracts their content, generates embeddings, and stores them in the database.
    
    Returns:
        int: Number of articles successfully scraped and stored
    """
    total_articles = 0
    page_url = CASE_STUDY_BASE_URL

    logger.info("🚀 CASE STUDY SCRAPER STARTED")

    while page_url:
        # Fetch page content
        response = requests.get(page_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all articles on the page
        articles = soup.find_all("article")
        if not articles:
            break

        for art in articles:
            # Extract article URL
            title_link = art.find("a", href=True)
            if not title_link:
                continue

            url = title_link["href"]

            # Check if article already exists in database
            cur.execute("SELECT 1 FROM dr_young_all_articles WHERE url=%s", (url,))
            if cur.fetchone():
                continue

            # Fetch detailed article content
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

            # Extract article title
            title_elem = detail_soup.find("h1")
            title = title_elem.get_text(strip=True) if title_elem else ""

            # Extract and clean article content
            content = extract_clean_article_content(detail_soup)
            if len(content) < MIN_CONTENT_LENGTH:
                continue

            # Generate embedding for semantic search
            embedding = model.encode(content).tolist()

            # Insert article into database
            cur.execute("""
                INSERT INTO dr_young_all_articles
                (title, url, content, embedding)
                VALUES (%s, %s, %s, %s)
            """, (title, url, content, str(embedding)))

            conn.commit()
            total_articles += 1
            logger.info(f"✅ INSERTED CASE STUDY: {title}")

            # Respectful delay between articles
            time.sleep(ARTICLE_DELAY)

        # Get next page URL for pagination
        next_link = soup.find("a", class_="next")
        page_url = urljoin(CASE_STUDY_BASE_URL, next_link["href"]) if next_link else None
        time.sleep(PAGE_DELAY)

    return total_articles


def scrape_dr_young_all_categories():
    """
    Scrape articles from all categories on Dr. Robert Young's website
    
    This function scrapes articles from multiple categories on drrobertyoung.com
    and stores them with embeddings for semantic search.
    
    Returns:
        int: Number of articles successfully scraped and stored
    """
    # Define categories to scrape
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
        # Construct category URL
        category_url = f"https://drrobertyoung.com/{category}/"

        try:
            # Fetch category page
            response = requests.get(
                category_url,
                headers=HEADERS,
                timeout=10,
                verify=False
            )
        except Exception:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all links on the page
        links = soup.find_all("a", href=True)
        post_urls = set()

        # Filter and collect valid post URLs
        for a in links:
            href = a["href"]

            # Only process drrobertyoung.com URLs
            if not href.startswith("https://drrobertyoung.com/"):
                continue

            # Skip unwanted URLs
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

            # Heuristic: likely articles have multiple hyphens in URL
            if href.count("-") < 3:
                continue

            post_urls.add(href)

        # Process each collected post URL
        for url in post_urls:
            # Check if article already exists in database
            cur.execute("SELECT 1 FROM dr_young_all_articles WHERE url=%s", (url,))
            if cur.fetchone():
                continue

            try:
                # Fetch detailed article content
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

            # Extract article title
            title_elem = soup.find("h1")
            title = title_elem.get_text(strip=True) if title_elem else url.split("/")[-1]

            # Extract and clean article content
            content = extract_clean_article_content(soup)
            if len(content) < MIN_CONTENT_LENGTH:
                continue

            # Generate embedding for semantic search
            embedding = model.encode(content).tolist()

            # Insert article into database
            cur.execute("""
                INSERT INTO dr_young_all_articles
                (title, url, content, embedding)
                VALUES (%s, %s, %s, %s)
            """, (title, url, content, str(embedding)))

            conn.commit()
            total_articles += 1
            logger.info(f"✅ INSERTED ARTICLE: {title}")

            # Respectful delay between articles
            time.sleep(ARTICLE_DELAY)

    return total_articles


# Main execution block
try:
    # Scrape case studies first
    cs = scrape_case_studies()
    time.sleep(3)  # Brief pause between scraping phases
    
    # Then scrape Dr. Robert Young's articles
    dr = scrape_dr_young_all_categories()

    # Log final statistics
    logger.info("🎉 SCRAPING COMPLETED")
    logger.info(f"Case Studies: {cs}")
    logger.info(f"Dr Young Articles: {dr}")
    logger.info(f"TOTAL: {cs + dr}")

finally:
    # Always close database connection
    conn.close()
