"""Bridge entrypoint for running from Frontend/ directory."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
