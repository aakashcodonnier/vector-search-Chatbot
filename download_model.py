#!/usr/bin/env python3
"""
Model downloader - Automatically downloads required LLM model
"""

import subprocess
import sys

def check_ollama():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def download_model(model_name="llama2"):  # Llama2 7B by default
    """Download specified model"""
    print(f"📥 Downloading {model_name} model...")
    print("This may take 10-15 minutes depending on your internet speed.")
    
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=False,  # Show real-time progress
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {model_name} downloaded successfully!")
            return True
        else:
            print(f"❌ Failed to download {model_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 40)
    print("🤖 MODEL DOWNLOADER")
    print("=" * 40)
    
    # Check Ollama
    if not check_ollama():
        print("❌ Ollama not found!")
        print("Please install Ollama first from: https://ollama.ai")
        return False
    
    print("✅ Ollama found")
    
    # Download model
    model = "llama2"  # Llama2 7B (7B params)
    
    # Optional: Let user choose model
    print("\nAvailable models:")
    print("1. llama2 (default, quality - 7B params)")
    print("2. mistral (balanced - 7B params)")
    print("3. tinyllama (fastest - 1.1B params)")
    
    choice = input("\nChoose model (1-3) or press Enter for default: ").strip()
    
    if choice == "2":
        model = "mistral"
    elif choice == "3":
        model = "tinyllama"
    # else default to llama2
    
    print(f"\nSelected model: {model}")
    
    if download_model(model):
        print(f"\n🎉 {model} is ready to use!")
        print("You can now run: python backend/main.py")
        return True
    else:
        print("\n⚠️  You can manually download later:")
        print(f"   ollama pull {model}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)