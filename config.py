"""Filesystem locations used by every script in the repository."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("PA_DATA_DIR", os.path.join(ROOT, "data"))
RESULTS_DIR = os.environ.get("PA_RESULTS_DIR", os.path.join(ROOT, "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)
