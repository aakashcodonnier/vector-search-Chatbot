from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from database.db import get_connection
import numpy as np
import ast

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")


class ChatRequest(BaseModel):
    question: str


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def clean_content(text: str) -> str:
    """
    Remove references / footer part from article
    """
    stop_words = [
        "References:",
        "REFERENCES:",
        "To learn more",
        "Click here",
        "For more information"
    ]
    for w in stop_words:
        if w in text:
            text = text.split(w)[0]
    return text.strip()


def summarize_text(text: str, max_sentences: int = 6) -> str:
    """
    Take only first 5–6 meaningful sentences
    """
    sentences = text.replace("\n", " ").split(". ")
    summary = ". ".join(sentences[:max_sentences])
    return summary.strip()


@app.post("/chat")
def chat(q: ChatRequest):
    query_emb = model.encode(q.question)

    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT title, content, embedding FROM articles")
    rows = cur.fetchall()

    best = []

    for r in rows:
        emb = np.array(ast.literal_eval(r["embedding"]))
        score = cosine(query_emb, emb)

        # 🔥 dynamic keyword boost
        if any(word in r["title"].lower() for word in q.question.lower().split()):
            score += 0.1

        if score > 0.30:
            best.append((score, r))

    best = sorted(best, key=lambda x: x[0], reverse=True)[:2]

    if not best:
        return {
            "answer": "No relevant article found in database",
            "references": []
        }

    references = []
    final_text = []

    for _, article in best:
        references.append(article["title"])

        cleaned = clean_content(article["content"])
        summary = summarize_text(cleaned, max_sentences=6)

        final_text.append(summary)

    return {
        "answer": "\n\n".join(final_text),
        "references": references
    }
