#!/usr/bin/env python3
"""
LSI Predictive AI Model - Dashboard Launcher
Simple script to launch the Streamlit dashboard
"""

import subprocess
import sys
import webbrowser
import time
import os

def main():
    print("=" * 60)
    print("🚀 LSI Predictive AI Model - Dashboard")
    print("=" * 60)
    print("\n📋 Starting the application...\n")
    
    # Change to the app directory
    os.chdir('/workspaces/AI_DASHBOARD')
    
    # Launch Streamlit
    try:
        # First, wait a moment for user to see the message
        time.sleep(1)
        
        print("✅ Application is starting...")
        print("\n📍 Access the dashboard at: http://localhost:8501")
        print("\n💡 The app will automatically open in your browser.")
        print("   If it doesn't, manually open: http://localhost:8501\n")
        print("🛑 Press Ctrl+C to stop the application\n")
        print("=" * 60 + "\n")
        
        # Launch the Streamlit app
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "lsi_streamlit_app.py"],
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard closed. Goodbye!")
    except FileNotFoundError:
        print("❌ Error: Could not find Streamlit. Please install it first:")
        print("   pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
