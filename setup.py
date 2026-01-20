#!/usr/bin/env python3
"""
Auto-setup script for the project
- Installs dependencies
- Downloads Ollama model
- Sets up database
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Executing: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print("✅ Success")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ Failed")
            if result.stderr:
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def main():
    print("=" * 50)
    print("🚀 PROJECT AUTO-SETUP")
    print("=" * 50)
    
    # 1. Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # 2. Install requirements
    if not run_command("pip install -r requirements.txt", 
                      "Installing Python dependencies"):
        return False
    
    # 3. Check/install Ollama
    print("\n🔍 Checking Ollama installation...")
    if not check_ollama_installed():
        print("❌ Ollama not found!")
        print("\nPlease install Ollama first:")
        print("1. Visit: https://ollama.ai")
        print("2. Download and install for Windows")
        print("3. Restart this script after installation")
        return False
    else:
        print("✅ Ollama found")
    
    # 4. Download model
    print("\n📥 Downloading LLM model...")
    if not run_command("ollama pull llama2", 
                      "Downloading llama2 model (may take 10-15 mins)"):
        print("⚠️  Model download failed. You can run this manually later:")
        print("   ollama pull llama2")
    
    # 5. Setup database instructions
    print("\n📋 DATABASE SETUP REQUIRED")
    print("Please create MySQL database with these settings:")
    print("- Database name: blog_qa")
    print("- Username: root")
    print("- Password: (your MySQL password)")
    print("- Host: localhost")
    print("- Port: 3306")
    
    print("\n📝 Update database/db.py with your credentials")
    
    # 6. Run scraper
    print("\n🕷️  Ready to scrape data?")
    choice = input("Run scraper now? (y/n): ").lower().strip()
    if choice == 'y':
        print("Starting scraper...")
        os.system("python scraper/scrape_and_embed.py")
    
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETE!")
    print("Next steps:")
    print("1. Update database credentials in database/db.py")
    print("2. Run: python backend/main.py")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)