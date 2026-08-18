"""
Streamlit Community Cloud entry point.
Imports and runs the dashboard implementation.
"""
import sys
from pathlib import Path

# Ensure the project root is in the Python path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import and run the dashboard
import app.views.dashboard