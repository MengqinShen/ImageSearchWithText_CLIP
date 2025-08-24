#!/usr/bin/env python3
"""
Simple launcher for the RAG Image Search Chatbot
"""

import sys
import os

def main():
    print("Starting RAG Image Search Chatbot...")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        "data_downloader.py",
        "data_processor.py", 
        "model_image_search.py",
        "gemini_ranker.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please make sure you're in the correct directory with all the RAG components.")
        return
    
    # Check if data directory exists
    if not os.path.exists("data"):
        print("❌ Data directory not found. Please run the main.py first to download data.")
        return
    
    print("✅ All required files found!")
    
    try:
        # Try to import and run the chatbot
        print("🔧 Importing chatbot...")
        from chatbot_simple import create_chatbot_interface
        
        print("Creating chatbot interface...")
        demo = create_chatbot_interface()
        
        print("Launching chatbot...")
        print("The chatbot will open in your browser at: http://localhost:7861")
        print("Press Ctrl+C to stop the chatbot")
        
        demo.launch(
            share=False, 
            server_name="0.0.0.0", 
            server_port=7861,
            show_error=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try installing dependencies with: pip install -r chatbot_requirements.txt")
        
    except Exception as e:
        print(f"❌ Error starting chatbot: {e}")
        print("💡 Check the console output for more details")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Chatbot stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}") 