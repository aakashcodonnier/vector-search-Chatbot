#!/usr/bin/env python3
"""
Unified Server for Dr. Robert Young Semantic Search

This server combines both the frontend (HTML/CSS/JS) and backend (API) 
into a single FastAPI application running on one port.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Import backend components
from backend.main import app as backend_app

# Create unified FastAPI app
app = FastAPI(
    title="Dr. Robert O . Young  - Unified Server",
    description="Combined frontend and backend server",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount backend routes under /api
app.mount("/api", backend_app)

# Serve static frontend files
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page"""
    frontend_path = frontend_dir / "index.html"
    if frontend_path.exists():
        with open(frontend_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    else:
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Please check the frontend directory.</p>",
            status_code=404
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "unified-server"}

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 UNIFIED SERVER STARTING")
    print("=" * 60)
    print("✨ Single URL for everything: http://192.168.1.43:8000")
    print("🌐 Frontend: http://127.0.0.1:8000/")
    print("🔌 Backend API: http://127.0.0.1:8000/api/")
    print("📚 API Docs: http://127.0.0.1:8000/api/docs")
    print("=" * 60)
    
    uvicorn.run(
        "unified_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )