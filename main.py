import os
import sys
import subprocess
import time
from pathlib import Path

def main():
    print("=" * 70)
    print("  Big Data Multimodal Fake News Detection & Propagation System")
    print("=" * 70)
    print("\nStarting system components...\n")

    # Ensure database directory exists
    os.makedirs("database", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("1. Launching Streamlit Glassmorphism Dashboard...")
    cmd = [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutdown complete.")

if __name__ == "__main__":
    main()
