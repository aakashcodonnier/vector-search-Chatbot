#!/usr/bin/env python3
"""
Backend API for Dr. Robert Young's semantic search Q&A system

This module provides a FastAPI application that:
1. Performs semantic search on scraped blog articles using vector embeddings
2. Generates contextual answers using local LLM via Ollama streaming
3. Maintains session-based conversation memory for context awareness
4. Provides detailed performance timing information for monitoring
5. Handles fallback responses gracefully when no relevant content found

Architecture Overview:
- Uses sentence-transformers for vector embeddings
- MySQL database stores articles with precomputed embeddings
- FastAPI serves REST API endpoints
- Ollama provides local LLM inference with streaming capability
- Session-based memory preserves conversation context across requests
"""

# Standard library imports
import sys          # System-specific parameters and functions
import os           # Operating system interface
import time         # Time access and conversions
import re           # Regular expression operations
import ast          # Abstract Syntax Tree processing
import json         # JSON encoder and decoder
import numpy as np  # Numerical computing library
from collections import deque  # Double-ended queue for efficient memory management

# Third-party imports
from fastapi import FastAPI                    # Web framework for API development
from fastapi.responses import StreamingResponse # Streaming response handler
from pydantic import BaseModel                 # Data validation and settings management
import requests                                # HTTP library for API calls
from sentence_transformers import SentenceTransformer  # Pre-trained transformer models

# Local imports
# Add the project root directory to the Python path for module resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from database.db import get_connection  # Database connection utility

# Initialize FastAPI app with metadata for API documentation
app = FastAPI(
    title="Dr. Robert Young Semantic Search API",
    description="Semantic search and Q&A system for Dr. Robert Young's blog content with session memory and streaming responses",
    version="1.0.0"
)

# Initialize embedding model for vector search operations
# Using all-MiniLM-L6-v2 for efficient sentence embeddings (22.7M parameters)
# This model provides good balance between speed and accuracy for semantic similarity
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Session-based conversation memory storage
# Dictionary mapping conversation_id to deque of recent interactions
# Each deque maintains last 5 interactions for optimal memory usage and context relevance
conversation_memory = {}

def get_conversation_history(conversation_id: str):
    """
    Get conversation history for given conversation ID
    
    This function retrieves or initializes conversation history for a specific session.
    Uses deque with maxlen=5 to automatically manage memory and keep only recent interactions.
    
    Args:
        conversation_id (str): Unique identifier for the conversation session
        
    Returns:
        deque: Double-ended queue containing conversation history items
    """
    if conversation_id not in conversation_memory:
        conversation_memory[conversation_id] = deque(maxlen=5)
    return conversation_memory[conversation_id]

def add_to_conversation_history(conversation_id: str, question: str, answer: str):
    """
    Add interaction to conversation history for context preservation
    
    This function stores question-answer pairs in the session memory to enable
    contextual follow-up questions and maintain conversation coherence.
    
    Args:
        conversation_id (str): Unique identifier for the conversation session
        question (str): User's question text
        answer (str): Generated answer text from the LLM
    """
    history = get_conversation_history(conversation_id)
    history.append({
        "question": question,
        "answer": answer,
        "timestamp": time.time()
    })
    
    # Log the addition for debugging and monitoring purposes
    print(f"💾 SAVED TO SESSION [{conversation_id}]:")
    print(f"   Question: {question[:60]}...")
    print(f"   Answer: {answer[:60]}...")
    print(f"   Total interactions: {len(history)}")

class ChatRequest(BaseModel):
    """
    Request model for chat endpoint with validation and defaults
    
    This Pydantic model defines the expected structure for incoming chat requests
    and provides automatic validation and serialization.
    
    Attributes:
        question (str): The user's question to be answered (required)
        conversation_id (str): Optional conversation identifier to maintain context (defaults to "default")
    """
    question: str
    conversation_id: str = "default"

def cosine(a, b):
    """
    Calculate cosine similarity between two vectors for semantic comparison
    
    Cosine similarity measures the cosine of the angle between two vectors,
    providing a normalized similarity score between 0 (completely different) 
    and 1 (identical) regardless of vector magnitude.
    
    Args:
        a (numpy.ndarray): First vector (typically query embedding)
        b (numpy.ndarray): Second vector (typically document embedding)
        
    Returns:
        float: Cosine similarity score between 0 and 1
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def clean_context(text: str) -> str:
    """
    Clean and preprocess context text for optimal LLM consumption
    
    This function removes unwanted formatting elements, reference sections,
    and other content that might confuse or distract the language model
    during response generation.
    
    Args:
        text (str): Raw text content to be cleaned
        
    Returns:
        str: Cleaned text ready for LLM processing
    """
    # Remove numbered points like "1.", "2)" that might interfere with LLM processing
    text = re.sub(r"\n?\s*\d+[\.]\)\s*", " ", text)

    # Remove bullet symbols and list markers that add no semantic value
    text = re.sub(r"[•\-–▪]", " ", text)

    # Remove reference/link sections that typically contain URLs and citations
    stop_words = ["References", "REFERENCES", "http", "www."]
    for w in stop_words:
        if w in text:
            text = text.split(w)[0]

    return text.strip()

# List of terms to avoid in responses to maintain neutrality and scientific accuracy
FORBIDDEN_TERMS = [
    "ph.d", "m.sc", "d.sc", "naturopath",      # Academic credentials
    "disseminated", "coagulation", "dic",       # Medical terminology
    "pathology", "mechanism", "theoretical",    # Technical/scientific terms
    "robert", "young"                           # Author names (avoid self-referential responses)
]

def sanitize_answer(text: str, question: str) -> str:
    """
    Sanitize answer based on question type for scientific accuracy and caution
    
    This function applies specific response patterns for certain types of questions
    to ensure scientifically accurate and appropriately cautious responses, particularly
    for questions about study validity, product comparisons, and health claims.
    
    Args:
        text (str): Original answer text from LLM
        question (str): Original user question
        
    Returns:
        str: Potentially modified answer based on question type and safety considerations
    """
    q = question.lower()

    # Case 1: Handle questions about study size or proof requirements
    # Ensures scientifically accurate responses about small study limitations
    if any(k in q for k in ["small", "three", "prove", "study"]):
        return (
            "Based on the information provided in the blog context, the study is exploratory "
            "and limited by its very small sample size. It cannot establish proof that the "
            "intervention removes toxins from the human body. The findings provide only initial "
            "observations, and larger, well-controlled studies would be required to draw "
            "definitive conclusions."
        )

    # Case 2: Handle questions about product differences or market comparisons
    # Ensures appropriate caution when comparing products or making claims
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
                    "num_predict": 100              # Maximum tokens to generate
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

@app.post("/chat")
async def chat(q: ChatRequest):
    """
    Main chat endpoint that processes user questions with session-based memory
    
    This endpoint performs semantic search on the blog database, maintains conversation
    context, and generates contextual answers using a local LLM.
    
    Args:
        q (ChatRequest): The user's question request with optional conversation ID
        
    Returns:
        dict: Response containing answer, references, and timing breakdown
    """
    # Start timing for performance measurement
    start_time = time.time()
    
    # Get conversation history
    history = get_conversation_history(q.conversation_id)
    
    # Log conversation tracking
    print(f"🔄 CONVERSATION SESSION: {q.conversation_id}")
    print(f"📊 HISTORY LENGTH: {len(history)} interactions")
    
    # Build context from conversation history
    history_context = ""
    if history:
        history_lines = []
        for item in history:
            history_lines.append(f"Previous question: {item['question']}")
            history_lines.append(f"Previous answer: {item['answer']}")
        history_context = "\n".join(history_lines) + "\n"
        print(f"📝 CONTEXT LINES ADDED: {len(history_lines)}")

    # 1️⃣ Embed user question using sentence transformer
    embed_start = time.time()
    query_emb = embed_model.encode(q.question)
    embed_time = time.time() - embed_start

    # 2️⃣ Fetch articles from database
    db_start = time.time()
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT title, content, embedding FROM dr_young_all_articles LIMIT 50")
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
        # Provide helpful general response instead of just saying "no info found"
        general_answer = (
            "While I don't have specific information about this topic in the available blog content, "
            "here are some general recommendations:\n\n"
            "• Minimize exposure when possible\n"
            "• Maintain distance from wireless devices\n"
            "• Focus on strengthening natural defenses\n"
            "• Stay informed about ongoing research"
        )
        
        return {
            "answer": general_answer,
            "references": ["General health guidance"],
            "timing": {
                "embedding": round(embed_time, 2),
                "database": round(db_time, 2),
                "search": round(search_time, 2),
                "context_building": 0.00,
                "llm_generation": 0.00,
                "total": round(time.time() - start_time, 2)
            }
        }

    # 4️⃣ Build context from top matching articles
    context_start = time.time()
    context_parts = []
    references = []  # Track source references

    for _, art in scored:
        # Add article title to references
        references.append(art["title"])
        # Clean and truncate article content
        cleaned = clean_context(art["content"][:200])
        context_parts.append(cleaned)

    print("CONTEXT LENGTH:", len(context_parts))

    # Join all context parts
    context = "\n\n".join(context_parts)
    context_time = time.time() - context_start

    # 5️⃣ Format prompt for LLM with conversation context
    prompt = f"""
{history_context}Context: {context}

Question: {q.question}

Answer (1-2 sentences max):"""

    llm_start = time.time()

    def stream_response():
        full_answer = ""

        for chunk in call_llama2_stream(prompt):
            full_answer += chunk
            yield chunk

        # Save conversation AFTER streaming finishes
        clean_answer = " ".join(full_answer.split())
        add_to_conversation_history(q.conversation_id, q.question, clean_answer)

        llm_time = time.time() - llm_start
        total_time = time.time() - start_time

        print("⏱️ TIMING:")
        print(f"Embedding: {embed_time:.2f}s | DB: {db_time:.2f}s | Search: {search_time:.2f}s")
        print(f"Context: {context_time:.2f}s | LLM: {llm_time:.2f}s | Total: {total_time:.2f}s")

    return StreamingResponse(
        stream_response(),
        media_type="text/plain"
    )
