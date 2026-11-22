"""Make every `rfproteina` module use the local vocabulary (rfproteina_encoding.py + ext_v2 yaml).

Import this FIRST in a script, before anything from rfproteina / rfd4_proteina:

    import local_vocab
    import rfd4_proteina as rfd
"""
import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "reference" / "rfd4_source" / "rfproteina_encoding.py"

_spec = importlib.util.spec_from_file_location("rfproteina.encoding", _PATH)
_module = importlib.util.module_from_spec(_spec)
sys.modules["rfproteina.encoding"] = _module  # later `from rfproteina.encoding import ...` finds this one
_spec.loader.exec_module(_module)
