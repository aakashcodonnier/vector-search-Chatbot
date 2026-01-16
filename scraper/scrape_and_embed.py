import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from database.db import get_connection

model = SentenceTransformer("all-MiniLM-L6-v2")

BASE_URL = "https://phoreveryoung.wordpress.com/category/case-studies/"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

conn = get_connection()
cur = conn.cursor()

# Table create
cur.execute("""
CREATE TABLE IF NOT EXISTS articles (
    id INT(11) NOT NULL AUTO_INCREMENT PRIMARY KEY,
    title TEXT,
    url TEXT,
    date VARCHAR(100),
    author VARCHAR(255),
    categories TEXT,
    content LONGTEXT,
    embedding LONGTEXT
)
""")

page = 1

while True:
    if page == 1:
        page_url = BASE_URL
    else:
        page_url = f"{BASE_URL}page/{page}/"

    print(f"\n📄 Scraping page: {page_url}")

    response = requests.get(page_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    articles = soup.find_all("article")
    if not articles:
        print("🚫 No more pages found. Stopping.")
        break

    for art in articles:
        title_tag = art.find("h1", class_="entry-title")
        if not title_tag or not title_tag.find("a"):
            continue

        url = title_tag.find("a")["href"]

        # 🔍 DUPLICATE CHECK
        cur.execute("SELECT id FROM articles WHERE url=%s LIMIT 1", (url,))
        if cur.fetchone():
            print(f"⏭️ Skipped duplicate: {url}")
            continue

        # Detail page
        detail_res = requests.get(url, headers=HEADERS)
        detail = BeautifulSoup(detail_res.text, "html.parser")

        title_el = detail.find("h1", class_="entry-title")
        title = title_el.get_text(strip=True) if title_el else ""

        date_el = detail.find("time", class_="entry-date")
        date = date_el.get_text(strip=True) if date_el else ""

        author_el = detail.find("span", class_="author")
        author = author_el.get_text(strip=True) if author_el else ""

        categories = ", ".join(
            c.get_text(strip=True)
            for c in detail.select(".cat-links a")
        )

        content_div = detail.find("div", class_="entry-content")
        if not content_div:
            continue

        content = " ".join(
            p.get_text(" ", strip=True)
            for p in content_div.find_all("p")
        )

        if len(content) < 200:
            continue

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
        print(f"✅ Inserted: {title}")

    page += 1

conn.close()
print("\n🎉 ALL CASE STUDY ARTICLES SCRAPED & STORED SUCCESSFULLY")
