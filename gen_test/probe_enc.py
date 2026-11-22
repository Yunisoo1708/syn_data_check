import os, sys
T = os.path.expanduser("~/rfd4_trial"); sys.path.insert(0, f"{T}/patch"); sys.path.insert(0, f"{T}/code")
import _common
import rfproteina_encoding_ext as X
E = X.UNIFIED_ATOM37_ENCODING
print([a for a in dir(E) if not a.startswith("_")][:60])
for n in ("DPN", "PHE", "DTR", "ALA"):
    i = E.token_to_idx[n]
    for att in ("idx_to_atom_names", "atom_names", "token_atom_names", "residue_atom_names"):
        if hasattr(E, att):
            v = getattr(E, att); 
            try: print(n, att, v[i] if not callable(v) else v(i))
            except Exception as ex: print(att, "err", ex)
for n in ("DPN", "PHE", "DTR", "ALA", "DAL"):
    ta = E.token_atoms; i = E.token_to_idx[n]
    try: print(n, "token_atoms", ta[i] if not callable(ta) else ta(i))
    except Exception as ex:
        try: print(n, "token_atoms[name]", ta[n])
        except Exception as ex2: print("err", ex, ex2)
print("atom_to_idx sample", list(E.atom_to_idx.items())[:6] if hasattr(E.atom_to_idx, "items") else type(E.atom_to_idx))
