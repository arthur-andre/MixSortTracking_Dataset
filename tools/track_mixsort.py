#!/usr/bin/env python3
"""Backward-compatible entry point. Prefer: python pipeline/tracking/run_batch.py"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "pipeline/tracking/run_batch.py"), run_name="__main__")
