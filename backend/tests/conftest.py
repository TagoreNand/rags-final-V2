"""
Shared pytest configuration.
Sets PYTHONPATH so tests import from project root.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
