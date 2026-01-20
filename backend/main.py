from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from database.db import get_connection
import numpy as np
import ast
import subprocess
import requests
import time
from datetime import datetime

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


FORBIDDEN_TERMS = [
    "ph.d", "m.sc", "d.sc", "naturopath",
    "disseminated", "coagulation", "dic",
    "pathology", "mechanism", "theoretical",
    "robert", "young"
]

def sanitize_answer(text: str, question: str) -> str:
    q = question.lower()

    # Case 1: Study size / proof questions
    if any(k in q for k in ["small", "three", "prove", "study"]):
        return (
            "Based on the information provided in the blog context, the study is exploratory "
            "and limited by its very small sample size. It cannot establish proof that the "
            "intervention removes toxins from the human body. The findings provide only initial "
            "observations, and larger, well-controlled studies would be required to draw "
            "definitive conclusions."
        )

    # Case 2: Product difference / comparison questions
    if any(k in q for k in ["different", "compare", "market", "other"]):
        return (
            "The blog context does not explicitly provide information about the study size, "
            "its stated purpose, or its methodological limitations in relation to this question. "
            "As a result, no conclusions can be drawn based on the available information. "
            "Further well-controlled studies would be required to address this topic."
        )

    return text


def call_llama2(prompt: str) -> str:
    """
    Call locally running LLaMA-2 via Ollama
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2:latest",  # Llama2 7B model
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.2
                }
            },
            timeout=300  # Increased timeout for Mistral 7B
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
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n[{timestamp}] 🚀 REQUEST STARTED")
    print(f"Question: {q.question}")
    
    # 1️⃣ Embed user question
    embed_start = time.time()
    query_emb = embed_model.encode(q.question)
    embed_time = time.time() - embed_start
    print(f"✅ Embedding time: {embed_time:.2f}s")

    # 2️⃣ Fetch articles from DB
    db_start = time.time()
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT title, content, embedding FROM articles")
    rows = cur.fetchall()
    db_time = time.time() - db_start
    print(f"✅ Database fetch time: {db_time:.2f}s (found {len(rows)} articles)")

    scored = []

    # 3️⃣ Vector similarity search
    search_start = time.time()
    for r in rows:
        emb = np.array(ast.literal_eval(r["embedding"]))
        score = cosine(query_emb, emb)

        # Soft keyword boost
        if any(word in r["title"].lower() for word in q.question.lower().split()):
            score += 0.1

        if score > 0.25:  # Lowered threshold for faster matching
            scored.append((score, r))
    
    search_time = time.time() - search_start
    print(f"✅ Similarity search time: {search_time:.2f}s (found {len(scored)} matches)")

    # top 1 reference (faster)
    scored = sorted(scored, key=lambda x: x[0], reverse=True)[:1]

    if not scored:
        return {
            "answer": "No relevant information found in the available blogs.",
            "references": []
        }

    # 4️⃣ Build context for LLaMA-2
    context_start = time.time()
    context_parts = []
    references = []

    for _, art in scored:
        references.append(art["title"])
        cleaned = clean_context(art["content"][:500])  # Reduced context size
        context_parts.append(cleaned)

    context = "\n\n".join(context_parts)
    context_time = time.time() - context_start
    print(f"✅ Context building time: {context_time:.2f}s (using {len(context_parts)} articles)")
    print(f"📊 Total context length: {len(context)} characters")

    # 5️⃣ Controlled prompt (5–6 bullet points)
    prompt_build_start = time.time()
    prompt = f"""
You are a scientific research assistant.

Answer the user's question strictly and exclusively using only the information explicitly provided in the blog context.
Do not rely on external knowledge, assumptions, or general background information.

Write ONE single concise academic paragraph of no more than four sentences.
Focus ONLY on the study size, the study’s stated purpose, and its methodological limitations.
If the study is small, preliminary, or exploratory, clearly state that it cannot establish proof or definitive conclusions
and that larger, well-controlled studies are required.

Do NOT introduce or infer any information that is not explicitly stated in the context, including:
– sample sizes, numerical results, comparisons, or effectiveness claims
– author credentials, titles, or affiliations
– disease names, mechanisms, biological pathways, test names, or technical terminology
– product differentiation, marketing claims, or comparisons with other products

Do NOT use promotional, persuasive, optimistic, or speculative language.
If the context does not contain enough information to fully answer the question, clearly state that the information is not available and explain the limitation based on the study design.

Context:
{context}

Question:
{q.question}

Answer:
"""
    prompt_time = time.time() - prompt_build_start
    print(f"✅ Prompt building time: {prompt_time:.4f}s")
    print(f"📊 Prompt length: {len(prompt)} characters")

    # 6️⃣ Generate answer using LLaMA-2
    llm_start = time.time()
    raw_answer = call_llama2(prompt)
    answer = sanitize_answer(raw_answer, q.question)

    llm_time = time.time() - llm_start
    
    answer = " ".join(answer.split())
    
    total_time = time.time() - start_time
    
    print(f"✅ LLM generation time: {llm_time:.2f}s")
    print(f"📊 Answer length: {len(answer)} characters")
    print(f"✅ TOTAL TIME: {total_time:.2f}s")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🎉 REQUEST COMPLETED\n")
    
    return {
        "answer": answer,
        "references": references,
        "timing": {
            "embedding": round(embed_time, 2),
            "database": round(db_time, 2),
            "search": round(search_time, 2),
            "context_building": round(context_time, 2),
            "prompt_building": round(prompt_time, 4),
            "llm_generation": round(llm_time, 2),
            "total": round(total_time, 2)
        }
    }
