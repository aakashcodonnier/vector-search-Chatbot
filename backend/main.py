from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from database.db import get_connection
import numpy as np
import ast
import subprocess
import requests

app = FastAPI()

# Embedding model (only for vector search)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


class ChatRequest(BaseModel):
    question: str


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

import re

def clean_context(text: str) -> str:
    # Remove numbered points like "1.", "2)"
    text = re.sub(r"\n?\s*\d+[\.\)]\s*", " ", text)

    # Remove bullet symbols
    text = re.sub(r"[•\-–▪]", " ", text)

    # Remove reference / links section
    stop_words = ["References", "REFERENCES", "http", "www."]
    for w in stop_words:
        if w in text:
            text = text.split(w)[0]

    return text.strip()



def call_llama2(prompt: str) -> str:
    """
    Call locally running LLaMA-2 via Ollama
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2:latest",
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )

        # 🔍 DEBUG: raw response
        print("OLLAMA STATUS:", response.status_code)
        print("OLLAMA RAW:", response.text)

        data = response.json()

        # ✅ SAFETY CHECKS
        if not isinstance(data, dict):
            return "Invalid response format from LLaMA-2."

        answer = data.get("response", "").strip()

        if not answer:
            return (
                "Based on the provided context, the available information is limited. "
                "The article discusses the topic in a general or theoretical manner "
                "and does not provide sufficient evidence to give a detailed answer."
            )

        return answer

    except Exception as e:
        return f"LLaMA-2 API error: {str(e)}"






@app.post("/chat")
def chat(q: ChatRequest):
    # 1️⃣ Embed user question
    query_emb = embed_model.encode(q.question)

    # 2️⃣ Fetch articles from DB
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT title, content, embedding FROM articles")
    rows = cur.fetchall()

    scored = []

    # 3️⃣ Vector similarity search
    for r in rows:
        emb = np.array(ast.literal_eval(r["embedding"]))
        score = cosine(query_emb, emb)

        # Soft keyword boost
        if any(word in r["title"].lower() for word in q.question.lower().split()):
            score += 0.1

        if score > 0.30:
            scored.append((score, r))

    # top 2 references
    scored = sorted(scored, key=lambda x: x[0], reverse=True)[:2]

    if not scored:
        return {
            "answer": "No relevant information found in the available blogs.",
            "references": []
        }

    # 4️⃣ Build context for LLaMA-2
    context_parts = []
    references = []

    for _, art in scored:
        references.append(art["title"])
        cleaned = clean_context(art["content"][:800])
        context_parts.append(cleaned)

    print("CONTEXT LENGTH:", len(context_parts))

    context = "\n\n".join(context_parts)

    # 5️⃣ Controlled prompt (5–6 bullet points)
    prompt = f"""
You are a scientific research assistant.

Answer the user's question strictly using only the information provided in the blog context.
Write ONE single concise academic paragraph of no more than 4 sentences.
Focus ONLY on the study size, its purpose, and its limitations.
Do NOT mention disease mechanisms, biological pathways, test names, or technical terminology.
Do NOT include multiple paragraphs or line breaks.
Do NOT use bullet points, numbering, or headings.
Do NOT use promotional, persuasive, or speculative language.
If the study is small or preliminary, clearly state that it provides only initial insights
and that larger, well-controlled studies are required.

Context:
{context}

Question:
{q.question}

Answer:
"""

    # 6️⃣ Generate answer using LLaMA-2
    answer = call_llama2(prompt)
    answer = " ".join(answer.split())

    return {
        "answer": answer,
        "references": references
    }
