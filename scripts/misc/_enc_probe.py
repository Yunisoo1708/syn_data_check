import os, sys
sys.path.insert(0, os.path.expanduser("~/rfd4_trial/patch")); sys.path.insert(0, os.path.expanduser("~/rfd4_trial/code"))
import _common
import rfd4_proteina as rfd
from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING as E
t2i = E.token_to_idx
print(len(t2i), list(t2i.items())[:5], list(t2i.items())[-12:])
for k in ["ALA", "VAL", "AIB", "DAL", "DTY", "UNK", "<M>", "<X>", "XCP", "MLE", "A", "DA", "GLY"]:
    print(k, t2i.get(k))
print({k: len(v) for k, v in __import__("rfproteina.encoding", fromlist=["x"]).RESIDUES_BY_TOKEN_TYPE.items()} if hasattr(__import__("rfproteina.encoding", fromlist=["x"]), "RESIDUES_BY_TOKEN_TYPE") else "")
