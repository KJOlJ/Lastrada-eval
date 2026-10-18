"""Multipage entrypoint for the experimental DTW / stem test bench."""

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "dtw_test_app.py"), run_name="__main__")
