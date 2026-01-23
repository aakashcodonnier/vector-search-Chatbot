#!/usr/bin/env python3
"""
Unified Server for Dr. Robert Young Semantic Search

This server combines both the frontend (HTML/CSS/JS) and backend (API) 
into a single FastAPI application running on one port, eliminating the need
for separate servers and CORS configuration.

Key Features:
- Serves frontend HTML/CSS/JS files directly
- Mounts backend API under /api prefix
- Handles CORS automatically
- Single deployment URL for entire application
- Health check endpoint for monitoring
"""

import os
import sys
from pathlib import Path

# Add project root to Python path for module imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# FastAPI framework imports
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles      # Serve static files
from fastapi.middleware.cors import CORSMiddleware  # Handle cross-origin requests
from fastapi.responses import HTMLResponse       # HTML response handling
import uvicorn                                   # ASGI server

# Import backend components from existing module
from backend.main import app as backend_app

# Create unified FastAPI app with descriptive metadata
app = FastAPI(
    title="Dr. Robert O . Young  - Unified Server",
    description="Combined frontend and backend server",
    version="1.0.0"
)

# Add CORS middleware to handle cross-origin requests
# This is essential for frontend-backend communication in development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins (adjust for production)
    allow_credentials=True,       # Allow cookies/credentials
    allow_methods=["*"],          # Allow all HTTP methods
    allow_headers=["*"]           # Allow all headers
)

# Mount backend routes under /api prefix
# This makes all backend endpoints available at /api/*
app.mount("/api", backend_app)

# Serve static frontend files from frontend directory
frontend_dir = Path(__file__).parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main frontend page
    
    This endpoint reads and serves the index.html file from the frontend directory,
    providing the complete chat interface to users.
    
    Returns:
        HTMLResponse: Complete frontend HTML page
    """
    frontend_path = frontend_dir / "index.html"
    if frontend_path.exists():
        with open(frontend_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    else:
        # Fallback error response if frontend files are missing
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Please check the frontend directory.</p>",
            status_code=404
        )

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and deployment verification
    
    This endpoint provides a simple way to verify that the server is running
    and responding correctly, useful for uptime monitoring and CI/CD pipelines.
    
    Returns:
        dict: Health status information
    """
    return {"status": "healthy", "service": "unified-server"}

if __name__ == "__main__":
    # Startup banner with connection information
    print("=" * 60)
    print("🚀 UNIFIED SERVER STARTING")
    print("=" * 60)
    print("✨ Single URL for everything: http://192.168.1.43:8000")
    print("🌐 Frontend: http://127.0.0.1:8000/")
    print("🔌 Backend API: http://127.0.0.1:8000/api/")
    print("📚 API Docs: http://127.0.0.1:8000/api/docs")
    print("=" * 60)
    
    # Start the server with auto-reload for development
    uvicorn.run(
        "unified_server:app",
        host="0.0.0.0",      # Listen on all interfaces
        port=8000,           # Standard development port
        reload=True          # Auto-reload on code changes
    )