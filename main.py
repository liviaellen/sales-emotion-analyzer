import argparse
import uvicorn
import streamlit.web.cli as stcli
import sys
import os

def run_fastapi():
    """Run the FastAPI application."""
    from app.fastapi_app import app
    uvicorn.run(app, host="0.0.0.0", port=8000)

def run_streamlit():
    """Run the Streamlit application."""
    sys.argv = ["streamlit", "run", "app/streamlit_app.py"]
    sys.exit(stcli.main())

def main():
    parser = argparse.ArgumentParser(description="Run the Sales Call Emotion Analyzer")
    parser.add_argument("--mode", choices=["fastapi", "streamlit"], default="streamlit",
                      help="Choose the application mode (default: streamlit)")
    args = parser.parse_args()

    if args.mode == "fastapi":
        print("Starting FastAPI server...")
        run_fastapi()
    else:
        print("Starting Streamlit interface...")
        run_streamlit()

if __name__ == "__main__":
    main()
