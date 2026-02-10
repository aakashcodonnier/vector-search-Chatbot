#!/usr/bin/env python3
"""Test script to diagnose Ollama connection issues"""

import requests
import json

def test_ollama_simple():
    """Test simple non-streaming request"""
    print("Testing simple request...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2:latest",
                "prompt": "What is zeolite?",
                "stream": False
            },
            timeout=30
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("[OK] Simple request works!")
            print(f"Response: {response.json().get('response', '')[:100]}...")
        else:
            print(f"[ERROR] Error: {response.text}")
    except Exception as e:
        print(f"[ERROR] Exception: {e}")

def test_ollama_streaming():
    """Test streaming request (matches app code)"""
    print("\nTesting streaming request (exact app format)...")
    try:
        with requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama2:latest",
                "prompt": "What is zeolite?",
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "repeat_penalty": 1.2,
                    "num_predict": 100
                }
            },
            stream=True,
            timeout=300
        ) as r:
            print(f"Status: {r.status_code}")

            if r.status_code != 200:
                print(f"[ERROR] Error: Ollama returned status {r.status_code}")
                print(f"Response text: {r.text}")
                return

            print("[OK] Streaming request works!")
            chunk_count = 0
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data and data["response"]:
                        chunk_count += 1
                        if chunk_count <= 5:  # Print first 5 chunks
                            print(f"  Chunk {chunk_count}: {data['response']}")
                    if data.get("done"):
                        print(f"[OK] Streaming completed! Total chunks: {chunk_count}")
                        break
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    continue

    except Exception as e:
        print(f"[ERROR] Exception: {e}")

def test_ollama_tags():
    """Test tags endpoint"""
    print("\nTesting tags endpoint...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"[OK] Found {len(models)} models")
            for model in models:
                print(f"  - {model['name']}")
        else:
            print(f"[ERROR] Error: {response.text}")
    except Exception as e:
        print(f"[ERROR] Exception: {e}")

if __name__ == "__main__":
    print("="*60)
    print("OLLAMA CONNECTION TEST")
    print("="*60)
    test_ollama_tags()
    test_ollama_simple()
    test_ollama_streaming()
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
