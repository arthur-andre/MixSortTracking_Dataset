"""Repository root helpers for pipeline scripts."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def setup_import_paths():
    """Ensure repo root and MixViT are on sys.path (call from pipeline/tracking scripts)."""
    for path in (REPO_ROOT, os.path.join(REPO_ROOT, "MixViT")):
        if path not in sys.path:
            sys.path.insert(0, path)
