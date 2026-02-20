#!/usr/bin/env python3
"""
Backend API for Dr. Robert Young's semantic search Q&A system

This module provides a FastAPI application that:
1. Performs semantic search on scraped blog articles
2. Generates contextual answers using local LLM
3. Provides performance timing information
"""

# Standard library imports
import sys
import os
import time
import re
import ast
import json
import subprocess
import numpy as np
from collections import deque
import asyncio

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Third-party imports
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
from sentence_transformers import SentenceTransformer

# Groq API for cloud LLM inference
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("[WARNING] Groq library not installed. Cloud deployment will not work.")

# Local imports
# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from database.db import get_connection

# Initialize FastAPI app
app = FastAPI(
    title="Dr. Robert Young Semantic Search API",
    description="Semantic search and Q&A system for Dr. Robert Young's blog content",
    version="1.0.0"
)

# Initialize embedding model for vector search
# Using all-MiniLM-L6-v2 for efficient sentence embeddings
import torch
from sentence_transformers import SentenceTransformer

device = "cpu"

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=device
)

# Initialize Groq client for cloud LLM (if API key available)
groq_client = None
if GROQ_AVAILABLE:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            groq_client = Groq(api_key=groq_api_key)
            print("[LLM] OK - Groq API initialized (Cloud mode)")
        except Exception as e:
            print(f"[LLM] ERROR - Groq initialization failed: {e}")
            groq_client = None
    else:
        print("[LLM] Using Ollama (Local mode - no GROQ_API_KEY found)")
else:
    print("[LLM] Using Ollama (Local mode - groq library not installed)")

# Session-based conversation memory (stores last 5 interactions per conversation)
conversation_memory = {}

def warm_up_ollama_model():
    """
    Warm up the Ollama model by making a simple request
    This pre-loads the model into memory to avoid delays on first user request
    """
    try:
        print("\n[OLLAMA] Warming up model...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2:latest",
                "prompt": "Hello",
                "stream": False
            },
            timeout=60
        )
        if response.status_code == 200:
            print("[OLLAMA] Model warmed up successfully!")
        else:
            print(f"[OLLAMA] Warm-up returned status {response.status_code}")
    except Exception as e:
        print(f"[OLLAMA] Warm-up failed: {e}")
        print("[OLLAMA] Model will load on first request (may take 10-15 seconds)")

# Warm up model on startup (run in background to not block server start)
import threading
threading.Thread(target=warm_up_ollama_model, daemon=True).start()

def get_conversation_history(conversation_id: str):
    """Get conversation history for given ID"""
    if conversation_id not in conversation_memory:
        conversation_memory[conversation_id] = deque(maxlen=5)
    return conversation_memory[conversation_id]

def add_to_conversation_history(conversation_id: str, question: str, answer: str):
    """Add interaction to conversation history"""
    history = get_conversation_history(conversation_id)
    history.append({
        "question": question,
        "answer": answer,
        "timestamp": time.time()
    })
    
    # Log the addition
    print(f"[SAVED] Session [{conversation_id}]:")
    print(f"   Question: {question[:60]}...")
    print(f"   Answer: {answer[:60]}...")
    print(f"   Total interactions: {len(history)}")


class ChatRequest(BaseModel):
    """
    Request model for chat endpoint
    
    Attributes:
        question (str): The user's question to be answered
        conversation_id (str): Optional conversation identifier to maintain context
    """
    question: str
    conversation_id: str = "default"


def cosine(a, b):
    """
    Calculate cosine similarity between two vectors
    
    Args:
        a (numpy.ndarray): First vector
        b (numpy.ndarray): Second vector
        
    Returns:
        float: Cosine similarity score between 0 and 1
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def clean_context(text: str) -> str:
    """
    Clean and preprocess context text for LLM consumption
    
    This function removes unwanted formatting elements that might confuse the LLM.
    
    Args:
        text (str): Raw text content to be cleaned
        
    Returns:
        str: Cleaned text ready for LLM processing
    """
    # Remove numbered points like "1.", "2)"
    text = re.sub(r"\n?\s*\d+[\.]\)\s*", " ", text)

    # Remove bullet symbols
    text = re.sub(r"[•\-–▪]", " ", text)

    # Remove standalone reference sections at the end (but keep inline URLs and references)
    text = re.sub(r"\n\s*References?\s*:\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)

    return text.strip()


# List of terms to avoid in responses to maintain neutrality
FORBIDDEN_TERMS = [
    "ph.d", "m.sc", "d.sc", "naturopath",
    "disseminated", "coagulation", "dic",
    "pathology", "mechanism", "theoretical",
    "robert", "young"
]


def sanitize_answer(text: str, question: str) -> str:
    """
    Sanitize answer based on question type
    
    This function applies specific response patterns for certain types of questions
    to ensure scientifically accurate and appropriately cautious responses.
    
    Args:
        text (str): Original answer text from LLM
        question (str): Original user question
        
    Returns:
        str: Potentially modified answer based on question type
    """
    q = question.lower()

    # Case 1: Handle questions about study size or proof
    if any(k in q for k in ["small", "three", "prove", "study"]):
        return (
            "Based on the information provided in the blog context, the study is exploratory "
            "and limited by its very small sample size. It cannot establish proof that the "
            "intervention removes toxins from the human body. The findings provide only initial "
            "observations, and larger, well-controlled studies would be required to draw "
            "definitive conclusions."
        )

    # Case 2: Handle questions about product differences or comparisons
    if any(k in q for k in ["different", "compare", "market", "other"]):
        return (
            "The blog context does not explicitly provide information about the study size, "
            "its stated purpose, or its methodological limitations in relation to this question. "
            "As a result, no conclusions can be drawn based on the available information. "
            "Further well-controlled studies would be required to address this topic."
        )

    return text


def call_llama2_stream(prompt: str):
    """
    Call locally running LLM via Ollama with streaming capability

    This function establishes a streaming connection to the Ollama service
    and yields response chunks as they become available, enabling real-time
    response delivery to the client.

    Args:
        prompt (str): Formatted prompt including context and question

    Yields:
        str: Response chunks from the LLM as they are generated

    Raises:
        Exception: If connection to Ollama fails or streaming encounters errors
    """
    try:
        # Test Ollama connection first
        test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if test_response.status_code != 200:
            yield "[LLM ERROR]: Ollama service not responding"
            return

        # Establish streaming POST request to Ollama API
        with requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2:latest",           # Specify LLaMA2 model
                "prompt": prompt,                   # Complete prompt with context
                "stream": True,                     # Enable streaming mode
                "options": {
                    "temperature": 0.7,             # Control randomness (0.7 = balanced creativity)
                    "top_p": 0.9,                   # Nucleus sampling parameter
                    "repeat_penalty": 1.2,          # Penalize repeated tokens
                    "num_predict": 300              # Maximum tokens to generate
                }
            },
            stream=True,                            # Enable response streaming
            timeout=300                             # 5-minute timeout for long responses
        ) as r:

            if r.status_code != 200:
                # Log detailed error for debugging
                error_text = r.text if hasattr(r, 'text') else 'No error details'
                print(f"[OLLAMA ERROR] Status {r.status_code}: {error_text}")
                yield f"[LLM ERROR]: Ollama returned status {r.status_code}. The model may be loading. Please wait a moment and try again."
                return

            # Track if we received any response
            received_response = False

            # Process streaming response line by line
            for line in r.iter_lines():
                if not line:
                    continue

                try:
                    # Parse JSON response chunk
                    data = json.loads(line.decode("utf-8"))

                    # Yield response content if available
                    if "response" in data and data["response"]:
                        received_response = True
                        yield data["response"]

                    # Stop streaming when generation is complete
                    if data.get("done"):
                        break

                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue
                except Exception as e:
                    yield f"[PARSING ERROR]: {str(e)}"
                    break

            # If no response was received, model might be loading
            if not received_response:
                yield "[LLM ERROR]: No response received. The model may still be loading. Please try again."

    except requests.exceptions.ConnectionError:
        yield "[LLM ERROR]: Cannot connect to Ollama service. Is it running? Start it with 'ollama serve'"
    except requests.exceptions.Timeout:
        yield "[LLM ERROR]: Ollama request timed out. The model may be loading or the prompt may be too complex."
    except Exception as e:
        # Handle streaming errors gracefully
        print(f"[OLLAMA EXCEPTION]: {type(e).__name__}: {str(e)}")
        yield f"[LLM ERROR]: {type(e).__name__} - {str(e)}"


def call_llama2_stream_direct(prompt: str):
    """
    Call locally running LLM via Ollama with streaming capability for direct responses

    This function establishes a streaming connection to the Ollama service
    and yields response chunks as they become available.

    Args:
        prompt (str): Formatted prompt including context and question

    Yields:
        str: Response chunks from the LLM as they are generated

    Raises:
        Exception: If connection to Ollama fails or streaming encounters errors
    """
    try:
        # Establish streaming POST request to Ollama API
        with requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2:latest",           # Specify LLaMA2 model
                "prompt": prompt,                   # Complete prompt with context
                "stream": True,                     # Enable streaming mode
                "options": {
                    "temperature": 0.7,             # Control randomness (0.7 = balanced creativity)
                    "top_p": 0.9,                   # Nucleus sampling parameter
                    "repeat_penalty": 1.2,          # Penalize repeated tokens
                    "num_predict": 300              # Maximum tokens to generate
                }
            },
            stream=True,                            # Enable response streaming
            timeout=300                             # 5-minute timeout for long responses
        ) as r:

            # Process streaming response line by line
            for line in r.iter_lines():
                if not line:
                    continue

                # Parse JSON response chunk
                data = json.loads(line.decode("utf-8"))

                # Yield response content if available
                if "response" in data:
                    yield data["response"]

                # Stop streaming when generation is complete
                if data.get("done"):
                    break

    except Exception as e:
        # Handle streaming errors gracefully
        yield f"\n[LLM ERROR]: {str(e)}"


def call_groq_stream(prompt: str):
    """
    Call Groq API with streaming capability (for cloud deployment)

    Groq provides ultra-fast LLM inference with streaming support.
    Uses llama-3.1-70b-versatile which is significantly more powerful
    and faster than llama2:7b.

    Benefits over Ollama:
    - 10x faster response time (1-2s vs 10-15s)
    - Better quality (70B vs 7B parameters)
    - No local resources needed
    - Perfect for Railway/cloud deployment

    Args:
        prompt (str): Formatted prompt including context and question

    Yields:
        str: Response chunks from the LLM as they are generated

    Raises:
        Exception: If Groq API fails or streaming encounters errors
    """
    try:
        if not groq_client:
            yield "[LLM ERROR]: Groq client not initialized. Check GROQ_API_KEY environment variable."
            return

        # Call Groq API with streaming
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You ARE Dr. Robert O. Young. You are NOT an AI assistant - you are Dr. Young himself answering questions.\n\n"
                        "CRITICAL RULES:\n"
                        "- NEVER say 'As an AI', 'Based on Dr. Young's research', 'According to Dr. Young', "
                        "'As I have emphasized in my research', or any phrase that reveals you are AI\n"
                        "- NEVER refer to yourself in third person. Do NOT say 'Dr. Young's work shows...' - "
                        "instead say 'My research shows...'\n"
                        "- Answer naturally as if you are the expert directly speaking to the person\n"
                        "- Speak with authority and confidence as the researcher who did this work\n\n"
                        "Your expertise: New Biology, pH Miracle lifestyle, cellular health, detoxification, "
                        "alkaline nutrition, structured water.\n\n"
                        "Key principles:\n"
                        "- Always answer based on the provided context\n"
                        "- Include specific chemical names with scientific abbreviations/formulas "
                        "(e.g., polyethylene glycol (PEG), graphene oxide (GO), ferric oxide (FeO))\n"
                        "- Include author names, journal names, and study titles exactly as in the context\n"
                        "- Include specific numbers, measurements, and scientific data from the context\n"
                        "- Reference MasterPeace Zeolite Z when relevant for detoxification\n"
                        "- Never make claims beyond what the provided content supports\n"
                        "- IMPORTANT: If the context contains URLs or links, include them at the end of your answer "
                        "with a short intro like 'See here for more info:' or 'Learn more here:' followed by the URLs, each on a new line"
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",  # Updated 70B model - much better than llama2:7B
            temperature=0.7,
            max_tokens=300,
            top_p=0.9,
            stream=True,
        )

        # Stream response chunks as they arrive
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                # Add small delay to make response feel more natural
                time.sleep(0.08)  # Increased to 80ms delay between chunks

    except Exception as e:
        yield f"[LLM ERROR]: {type(e).__name__} - {str(e)}"


def call_groq_stream_direct(prompt: str):
    """
    Call Groq API with streaming for direct responses (Case 3 fallback)

    Same as call_groq_stream but optimized for direct answer generation
    without database context.

    Args:
        prompt (str): Formatted prompt

    Yields:
        str: Response chunks from the LLM
    """
    try:
        if not groq_client:
            yield "[LLM ERROR]: Groq client not initialized"
            return

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You ARE Dr. Robert O. Young. Never say 'As an AI' or refer to yourself in third person. "
                        "Answer naturally as the expert. Your expertise: New Biology, pH Miracle lifestyle, "
                        "alkaline health, cellular detoxification. Include specific chemical names, "
                        "scientific data, and study references. Speak with authority."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=300,
            top_p=0.9,
            stream=True,
        )

        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                # Add small delay to make response feel more natural
                time.sleep(0.08)  # Increased to 80ms delay between chunks

    except Exception as e:
        yield f"\n[LLM ERROR]: {type(e).__name__} - {str(e)}"


@app.post("/chat")
async def chat(q: ChatRequest):
    """
    Main chat endpoint that processes user questions with session-based memory
    
    This endpoint performs semantic search on the blog database, maintains conversation
    context, and generates contextual answers using a local LLM.
    
    Args:
        q (ChatRequest): The user's question request with optional conversation ID
        
    Returns:
        StreamingResponse: Streaming response containing answer and references
    """
    # Start timing for performance measurement
    start_time = time.time()
    
    # Get conversation history
    history = get_conversation_history(q.conversation_id)
    
    # Log conversation tracking
    print(f"[SESSION] CONVERSATION: {q.conversation_id}")
    print(f"[HISTORY] LENGTH: {len(history)} interactions")
    
    # Build context from conversation history
    history_context = ""
    if history:
        history_lines = []
        for item in history:
            history_lines.append(f"Previous question: {item['question']}")
            history_lines.append(f"Previous answer: {item['answer']}")
        history_context = "\n".join(history_lines) + "\n"
        print(f"[CONTEXT] LINES ADDED: {len(history_lines)}")

    # 1️⃣ Embed user question using sentence transformer
    embed_start = time.time()
    query_emb = embed_model.encode(
        q.question,
        convert_to_numpy=True,
        device="cpu"
    )
    embed_time = time.time() - embed_start

    # 2️⃣ Fetch articles from database
    db_start = time.time()
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT title, content, embedding, url FROM dr_young_all_articles")
    rows = cur.fetchall()
    db_time = time.time() - db_start

    # Prepare list to store similarity scores
    scored = []

    # 3️⃣ Perform vector similarity search
    search_start = time.time()
    for r in rows:
        # Convert stored embedding string back to numpy array
        emb = np.array(ast.literal_eval(r["embedding"]))
        # Calculate cosine similarity with query
        score = cosine(query_emb, emb)

        # Boost score if query terms appear in title
        if any(word in r["title"].lower() for word in q.question.lower().split()):
            score += 0.1

        # Only consider results above threshold
        if score > 0.30:
            scored.append((score, r))

    # Sort by similarity score and take top result
    scored = sorted(scored, key=lambda x: x[0], reverse=True)[:1]
    search_time = time.time() - search_start

    # Return if no relevant results found
    if not scored:
        # Generate contextually appropriate response based on question type and conversation context
        question_lower = q.question.lower()
        
        # Case 4: No DB results and no conversation context - Polite refusal
        if not history:
            print(f"[CASE 4] TRIGGERED: No DB match + No conversation history for '{q.question}'")
            general_answer = (
                "I don't have reliable information about this specific topic in the available content. "
                "Could you please provide more details or rephrase your question? "
                "Alternatively, you might want to ask about related topics like general health principles, "
                "wellness practices, or preventive care approaches."
            )
        
        # Case 3: No DB results but continuing previous conversation topic
        elif history:
            # Check if current question relates to previous conversation topics
            # MATCH AGAINST BOTH QUESTION AND ANSWER (as per memory requirement)
            last_interaction = history[-1] if history else {}
            last_question = last_interaction.get('question', '').lower() if last_interaction else ""
            last_answer = last_interaction.get('answer', '').lower() if last_interaction else ""
            
            # Combine question and answer for better context matching
            combined_context = f"{last_question} {last_answer}"
            conversation_keywords = set(combined_context.split())
            current_keywords = set(question_lower.split())
            
            # Calculate keyword overlap
            overlap = len(conversation_keywords.intersection(current_keywords))
            total_unique = len(conversation_keywords.union(current_keywords))
            similarity_ratio = overlap / total_unique if total_unique > 0 else 0
            
            # If there's significant topic continuity (30%+ keyword overlap)
            if similarity_ratio > 0.3 or any(word in question_lower for word in combined_context.split()[:10]):
                print(f"[CASE 3] TRIGGERED: No DB match + Continuing topic ({similarity_ratio:.2f} similarity) for '{q.question}'")
                print(f"   Matching against: Question='{last_question[:50]}...' Answer='{last_answer[:50]}...'")
                
                # CASE 3 LLM FALLBACK - Generate answer using LLM for topic continuity
                # Build context from conversation history for LLM prompt
                llm_context = f"Based on our previous discussion about: {last_question}\nPrevious answer context: {last_answer[:200]}"
                
                llm_prompt = f"""
{llm_context}

Question: {q.question}

Answer this question directly as Dr. Robert O. Young. Do NOT say "As an AI" or refer to yourself in third person. Speak naturally and confidently as the expert. Provide a detailed, informative answer (3-5 sentences).
"""
                
                # Generate answer using LLM for Case 3
                llm_start = time.time()
                
                def stream_case3_response():
                    full_answer = ""

                    # Stream the prefix immediately
                    yield "Answer With AI:\n\n"
                    
                    # Stream the LLM response chunks as they arrive
                    try:
                        # Auto-select: Use Groq if available (cloud), otherwise Ollama (local)
                        llm_function = call_groq_stream_direct if groq_client else call_llama2_stream_direct
                        for chunk in llm_function(llm_prompt):
                            if chunk.strip():  # Only yield non-empty chunks
                                full_answer += chunk
                                yield chunk
                                # Add small delay to make response feel more natural
                                time.sleep(0.05)  # Increased to 50ms delay between frontend updates
                                
                        # Save conversation AFTER streaming finishes
                        clean_answer = " ".join(full_answer.split())
                        add_to_conversation_history(q.conversation_id, q.question, clean_answer)
                        
                        llm_time = time.time() - llm_start
                        print(f"[TIMING] CASE 3 LLM: {llm_time:.2f}s")
                    except Exception as e:
                        yield f"[LLM ERROR]: {str(e)}"
                
                return StreamingResponse(stream_case3_response(), media_type="text/plain")
                
            else:
                print(f"[CASE 4] TRIGGERED: No DB match + Different topic ({similarity_ratio:.2f} similarity) for '{q.question}'")
                print(f"   Context: Question='{last_question[:50]}...' Answer='{last_answer[:50]}...'")
                # Different topic - Case 4 handling
                general_answer = (
                    "I don't have reliable information about this specific topic in the available content. "
                    "Could you please provide more details or rephrase your question? "
                    "Alternatively, you might want to ask about related topics like general health principles, "
                    "wellness practices, or preventive care approaches."
                )
        
        def stream_general_response():
            yield general_answer
        
        return StreamingResponse(stream_general_response(), media_type="text/plain")

    # 4️⃣ Build context from top matching articles
    context_start = time.time()
    context_parts = []
    references = []  # Track source references with URLs

    for _, art in scored:
        # Add article title and URL to references
        references.append({
            "title": art["title"],
            "url": art["url"]
        })
        # Clean and truncate article content
        cleaned = clean_context(art["content"][:1500])
        context_parts.append(cleaned)

    print("CONTEXT LENGTH:", len(context_parts))

    # Join all context parts
    context = "\n\n".join(context_parts)
    context_time = time.time() - context_start

    # 5️⃣ Format prompt for LLM with conversation context
    prompt = f"""
{history_context}The following is from your published articles and research:

{context}

Answer the following question directly as yourself. Do NOT say "As an AI" or "According to my research" - just answer naturally and confidently. Include specific chemical names with abbreviations, scientific data, and study references exactly as they appear above. Provide a detailed, informative answer.

IMPORTANT: If the context contains any URLs or links, you MUST include them at the end of your answer. Write a short intro line like "See here for more info:" or "Learn more here:" followed by the actual URLs from the context, each on a new line. Do NOT skip any URLs from the context.

Question: {q.question}

Answer:"""

    llm_start = time.time()

    def stream_response():

        full_answer = ""

        # Stream the prefix immediately
        yield "Answer With AI:\n\n"

        # Stream the LLM response chunks as they arrive
        try:
            # Auto-select: Use Groq if available (cloud), otherwise Ollama (local)
            llm_function = call_groq_stream if groq_client else call_llama2_stream
            for chunk in llm_function(prompt):
                if chunk.strip():  # Only yield non-empty chunks
                    full_answer += chunk
                    yield chunk
                    # Add small delay to make response feel more natural
                    time.sleep(0.05)  # Increased to 50ms delay between frontend updates
                    
            # Add references with article titles
            yield "\n\nReferences:\n"
            for i, ref in enumerate(references, 1):
                yield f"{i}. {ref['title']}\n"

            # Save conversation AFTER streaming finishes
            clean_answer = " ".join(full_answer.split())
            add_to_conversation_history(q.conversation_id, q.question, clean_answer)
            
            llm_time = time.time() - llm_start
            total_time = time.time() - start_time
            
            print("[TIMING]:")
            print(f"Embedding: {embed_time:.2f}s | DB: {db_time:.2f}s | Search: {search_time:.2f}s")
            print(f"Context: {context_time:.2f}s | LLM: {llm_time:.2f}s | Total: {total_time:.2f}s")
        except Exception as e:
            yield f"[LLM ERROR]: {str(e)}"

    return StreamingResponse(
        stream_response(),
        media_type="text/plain"
    )
