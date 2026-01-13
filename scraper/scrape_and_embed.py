import requests
import json
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from database.db import get_connection

BASE_URL = "https://phoreveryoung.wordpress.com/category/case-studies/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# 🔹 Load embedding model (vector generator)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 DB connection
conn = get_connection()
cursor = conn.cursor()

page = 1
total_inserted = 0

while True:
    page_url = f"{BASE_URL}page/{page}/"
    print(f"🔍 Scraping page {page}: {page_url}")

    response = requests.get(page_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    articles = soup.find_all("article")

    # ❌ No more pages
    if not articles:
        print("🚫 No more articles found. Stopping scraping.")
        break

    for art in articles:
        # 🔹 Title + URL
        title_tag = art.find("h1", class_="entry-title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        url = title_tag.find("a")["href"] if title_tag and title_tag.find("a") else ""

        if not url:
            continue

        # 🔹 Duplicate check (VERY IMPORTANT)
        cursor.execute("SELECT id FROM articles WHERE url = %s", (url,))
        if cursor.fetchone():
            print(f"⏩ Skipping duplicate: {title}")
            continue

        # 🔹 Date
        date_tag = art.find("time")
        date = date_tag.get_text(strip=True) if date_tag else ""

        # 🔹 Author
        author_tag = art.find("span", class_="author")
        author = author_tag.get_text(strip=True) if author_tag else ""

        # 🔹 Categories
        categories = ", ".join(
            c.get_text(strip=True) for c in art.select(".cat-links a")
        )

        # 🔹 Content (full text visible on category page)
        content_div = art.find("div", class_="entry-content")
        content_text = content_div.get_text(" ", strip=True) if content_div else ""

        if not content_text:
            continue

        # 🔹 Create vector embedding
        embedding = model.encode(title + " " + content_text).tolist()

        # 🔹 Insert into SQL DB
        cursor.execute(
            """
            INSERT INTO articles 
            (title, url, date, author, categories, content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                title,
                url,
                date,
                author,
                categories,
                content_text,
                json.dumps(embedding)
            )
        )

        total_inserted += 1
        print(f"✅ Inserted: {title}")

    conn.commit()
    page += 1

cursor.close()
conn.close()

print(f"\n🎉 DONE! Total articles inserted: {total_inserted}")
