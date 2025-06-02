#!/usr/bin/env python
# coding: utf-8

"""
Launch script for Arabic Text Classification Demo
Checks dependencies and starts the Streamlit application
"""

import subprocess
import sys
import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if required file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: Found")
        return True
    else:
        print(f"❌ {description}: Missing - {filepath}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit', 'torch', 'transformers', 'scikit-learn', 
        'pandas', 'plotly', 'nltk', 'arabert'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Missing")
            missing_packages.append(package)
    
    return missing_packages

def check_required_files():
    """Check if all required model files exist"""
    required_files = [
        ("bilstm_best_model.pth", "BiLSTM model"),
        ("results/checkpoint-288-best", "AraBERT model directory"),
        ("processed_classification_data.csv", "Classification data"),
        ("processed_ngram_data.csv", "N-gram data"),
        ("src/models/traditional.py", "Traditional models"),
        ("src/models/bilstm.py", "BiLSTM model code"),
        ("src/models/arabert.py", "AraBERT model code"),
        ("src/preprocessing/text_preprocessing.py", "Preprocessing code")
    ]
    
    all_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist

def install_missing_packages(packages):
    """Install missing packages"""
    if packages:
        print(f"\n🔧 Installing missing packages: {', '.join(packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
            print("✅ Packages installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"pip install {' '.join(packages)}")
            return False
    return True

def launch_streamlit():
    """Launch the Streamlit application"""
    print("\n🚀 Launching Arabic Text Classification Demo...")
    print("📱 The app will open in your default browser")
    print("🌐 URL: http://localhost:8501")
    print("⚠️  Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to launch Streamlit. Try running manually:")
        print("streamlit run app.py")
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")

def main():
    """Main function"""
    print("🔤 Arabic Text Classification Demo - Launch Script")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ app.py not found. Please run this script from the project root directory.")
        return
    
    print("\n📋 Checking dependencies...")
    missing_packages = check_dependencies()
    
    print("\n📁 Checking required files...")
    files_exist = check_required_files()
    
    if missing_packages:
        print(f"\n⚠️  Missing packages detected: {', '.join(missing_packages)}")
        install = input("Install missing packages? (y/n): ").lower().strip()
        if install == 'y':
            if not install_missing_packages(missing_packages):
                return
        else:
            print("❌ Cannot proceed without required packages.")
            return
    
    if not files_exist:
        print("\n❌ Some required files are missing. Please ensure:")
        print("   • Model files are in place (bilstm_best_model.pth, results/checkpoint-288-best)")
        print("   • Data files are processed (processed_*.csv)")
        print("   • Source code is organized in src/ directory")
        return
    
    print("\n✅ All checks passed!")
    launch_streamlit()

if __name__ == "__main__":
    main() 