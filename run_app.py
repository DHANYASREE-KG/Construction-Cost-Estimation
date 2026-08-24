"""
========================================================================================
BuildCost AI - Full-Stack App Launcher (Backend + Frontend)
========================================================================================
Runs the FastAPI server hosting the REST API and the interactive Web Dashboard.
"""
import os
import sys
import webbrowser
import uvicorn

if __name__ == "__main__":
    # Ensure current directory is in sys.path
    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "backend")
    sys.path.insert(0, app_dir)
    
    url = "http://127.0.0.1:8000"
    print("=" * 70)
    print("      BUILDCOST AI - FUTURE BUILDING COST ESTIMATION PLATFORM")
    print("=" * 70)
    print(f" -> Backend API: {url}/docs")
    print(f" -> Web Dashboard: {url}")
    print("=" * 70)
    
    # Auto-open browser
    try:
        webbrowser.open(url)
    except Exception:
        pass
        
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, app_dir=app_dir)
