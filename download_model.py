#!/usr/bin/env python3
"""
Model Downloader for Dr. Robert Young's Semantic Search System

This module automatically downloads the required LLM model for the semantic search system.
It checks for Ollama installation and pulls the specified model.
"""

# Standard library imports
import subprocess
import sys


def check_ollama():
    """
    Check if Ollama is installed and available
    
    Returns:
        bool: True if Ollama is available, False otherwise
    """
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def download_model(model_name="mistral"):  # Mistral (better quality, faster)
    """
    Download the specified LLM model using Ollama
    
    Args:
        model_name (str): Name of the model to download
        
    Returns:
        bool: True if download successful, False otherwise
    """
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
    """
    Main function to handle the model download process
    
    This function checks for Ollama, presents model options to the user,
    and downloads the selected model.
    
    Returns:
        bool: True if download successful, False otherwise
    """
    print("=" * 40)
    print("🤖 MODEL DOWNLOADER")
    print("=" * 40)
    
    # Check if Ollama is installed
    if not check_ollama():
        print("❌ Ollama not found!")
        print("Please install Ollama first from: https://ollama.ai")
        return False
    
    print("✅ Ollama found")
    
    # Set default model
    model = "mistral"  # Mistral (better quality, faster)
    
    # Present model options to user
    print("\nAvailable models:")
    print("1. llama2 (default, quality - 7B params)")
    print("2. mistral (recommended - better quality, faster)")
    print("3. tinyllama (fastest - 1.1B params)")
    
    choice = input("\nChoose model (1-3) or press Enter for mistral (recommended): ").strip()
    
    if choice == "1":
        model = "llama2"
    elif choice == "3":
        model = "tinyllama"
    # else default to mistral
    
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