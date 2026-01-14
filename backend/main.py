from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json, requests
from database.db import get_connection

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

class QuestionRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(data: QuestionRequest):
    question = data.question

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    q_vec = model.encode(question)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT title, content, embedding FROM articles")
    rows = cursor.fetchall()

    scored = []
    for r in rows:
        score = cosine_similarity(
            [q_vec],
            [json.loads(r["embedding"])]
        )[0][0]
        scored.append((score, r))

    top = sorted(scored, reverse=True)[:2]

    context = ""
    references = []

    for _, art in top:
        context += art["content"][:600] + "\n\n"
        references.append(art["title"])

    prompt = f"""
    You are a research assistant.
    Answer ONLY using the data below.
    If not found, say "Not found in database".

    DATA:
    {context}

    QUESTION:
    {question}
    """

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        answer = res.json().get("response", "No response from model")

    except Exception as e:
        answer = f"Ollama error: {str(e)}"



    return {
        "answer": answer.strip(),
        "references": references
    }
