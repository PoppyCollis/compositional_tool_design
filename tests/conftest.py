"""Put the repo root on sys.path so the flat top-level modules (tool_geometry,
config, ...) import under a bare `pytest` invocation, not just `python -m pytest`
(which prepends the cwd for you)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
