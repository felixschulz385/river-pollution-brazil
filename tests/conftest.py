from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# VS Code pytest discovery can import the stdlib `code` module before the
# workspace package is visible on sys.path. Drop that cached module so later
# imports resolve to the repository package at `PROJECT_ROOT/code`.
loaded_code_module = sys.modules.get("code")
if loaded_code_module is not None and not hasattr(loaded_code_module, "__path__"):
    sys.modules.pop("code", None)
