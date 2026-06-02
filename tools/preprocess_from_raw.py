#!/usr/bin/env python3
"""Backward-compatible entry point. Prefer: python pipeline/preprocess/from_raw.py"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "pipeline/preprocess/from_raw.py"), run_name="__main__")
