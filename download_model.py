#!/usr/bin/env python3
"""
Model Downloader for Dr. Robert Young's Semantic Search System



This module automatically downloads the required LLM model for the semantic search system.
It checks for Ollama installation and pulls the specified model.

Key Features:
- Automatic Ollama installation verification
- Interactive model selection interface
- Cross-platform support (Windows, macOS, Linux)
- Browser integration for easy downloads
- Progress reporting during model downloads

Supported Models:
1. llama2 - Default quality model (7B parameters)
2. mistral - Recommended model (better quality, faster)
3. tinyllama - Fastest model (1.1B parameters)

Requirements:
- Ollama must be installed and accessible from command line
- Internet connection for model downloads
- Sufficient disk space for model storage
"""

# Standard library imports
import subprocess    # Execute system commands for Ollama interaction
import sys           # System-specific parameters and functions
import platform      # Platform identification for OS-specific handling
import webbrowser    # Web browser integration for download assistance


def check_ollama():
    """
    Check if Ollama is installed and available in system PATH
    
    This function verifies that Ollama is properly installed and accessible
    by attempting to execute the 'ollama --version' command.
    
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
    
    This function initiates the download process for the requested model
    through Ollama's command-line interface, showing real-time progress.
    
    Args:
        model_name (str): Name of the model to download (default: "mistral")
        
    Returns:
        bool: True if download successful, False otherwise
    """
    print(f"📥 Downloading {model_name} model...")
    print("This may take 10-15 minutes depending on your internet speed.")
    
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=False,  # Show real-time progress to user
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


def print_installation_guide():
    """
    Print detailed installation guide for Ollama with platform-specific instructions
    
    This function provides comprehensive installation instructions based on
    the detected operating system, making it easy for users to install Ollama.
    """
    print("\n" + "=" * 50)
    print("🔧 OLLAMA INSTALLATION REQUIRED")
    print("=" * 50)
    print("\nTo use this semantic search system, you need to install Ollama first.")
    
    # Detect operating system for platform-specific instructions
    system = platform.system().lower()
    if system == "windows":
        print("\n🖥️  WINDOWS USERS:")
        print("1. Visit: https://ollama.ai/download/windows")
        print("2. Download the Windows installer (.exe file)")
        print("3. Run the installer as Administrator")
        print("4. Follow the setup wizard")
    elif system == "darwin":
        print("\n🍎 MAC USERS:")
        print("1. Visit: https://ollama.ai/download/mac")
        print("2. Download the macOS installer")
        print("3. Open the downloaded file and drag Ollama to Applications")
    else:
        print("\n🐧 LINUX USERS:")
        print("1. Visit: https://ollama.ai/download/linux")
        print("2. Follow the curl installation command provided")
    
    print("\n📋 GENERAL STEPS:")
    print("4. After installation, restart your terminal/command prompt")
    print("5. Run this script again: python download_model.py")
    
    print("\n💡 TIP: After installing Ollama, verify it's working:")
    print("   ollama --version")
    
    # Offer to open download page in browser for convenience
    try:
        choice = input("\nWould you like me to open the Ollama download page? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            webbrowser.open("https://ollama.ai/download")
            print("✅ Opening Ollama download page in your browser...")
    except:
        pass  # Silently continue if webbrowser fails
    
    print("\n" + "=" * 50)


def main():
    """
    Main function to handle the model download process
    
    This function orchestrates the complete model setup workflow:
    1. Checks for Ollama installation
    2. Presents model options to the user
    3. Downloads the selected model
    4. Provides success/failure feedback
    
    Returns:
        bool: True if download successful, False otherwise
    """
    print("=" * 40)
    print("🤖 MODEL DOWNLOADER")
    print("=" * 40)
    
    # Check if Ollama is installed before proceeding
    if not check_ollama():
        print("❌ OLLAMA NOT FOUND")
        print("This system requires Ollama to run local LLM models.")
        print_installation_guide()
        return False
    
    print("✅ Ollama found and ready!")
    
    # Set default model recommendation
    model = "mistral"  # Mistral (better quality, faster)
    
    # Present model options to user with descriptions
    print("\nAvailable models:")
    print("1. llama2 (default, quality - 7B params)")
    print("2. mistral (recommended - better quality, faster)")
    print("3. tinyllama (fastest - 1.1B params)")
    
    choice = input("\nChoose model (1-3) or press Enter for mistral (recommended): ").strip()
    
    # Process user choice
    if choice == "1":
        model = "llama2"
    elif choice == "3":
        model = "tinyllama"
    # else default to mistral
    
    print(f"\nSelected model: {model}")
    
    # Attempt model download and report results
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