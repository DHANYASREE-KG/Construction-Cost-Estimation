"""
========================================================================================
BuildCost AI - 1-Click Application Launcher (Backend + Frontend)
========================================================================================
"""
import os
import sys
import webbrowser
import uvicorn

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(current_dir, "backend")
    sys.path.insert(0, backend_dir)
    
    url = "http://127.0.0.1:8000"
    print("=" * 75)
    print("      BUILDCOST AI - FUTURE BUILDING CONSTRUCTION COST ESTIMATOR")
    print("=" * 75)
    print(f" -> Web Dashboard: {url}")
    print(f" -> Swagger API Docs: {url}/docs")
    print("=" * 75)
    print("Starting server... (Press Ctrl+C to stop)")
    
    try:
        webbrowser.open(url)
    except Exception:
        pass
        
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, app_dir=backend_dir)
