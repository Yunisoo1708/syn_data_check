"""Probe: where does the NCAA identity enter generation when the spec has NO C_SEQ on the peptide (spec_free.yaml)?
Wraps autoencoder.decode and prints, at the peptide tokens, the C_SEQ feature (identity given to the decoder, <M> = hidden), the input residue_type, the token_type and the decoder's own argmax.
usage (box): python probe_identity.py [EXAMPLE=7PIU_P] [SPEC=spec_free.yaml]"""
import os, sys
T = os.path.expanduser("~/rfd4_trial")
sys.path.insert(0, f"{T}/patch"); sys.path.insert(0, f"{T}/code")
import _common
import torch, rfd4_proteina as rfd
from atomworks.ml.conditions import C_SEQ
import rfproteina_encoding_ext as X
E = X.UNIFIED_ATOM37_ENCODING
ex = sys.argv[1] if len(sys.argv) > 1 else "7PIU_P"; spec = sys.argv[2] if len(sys.argv) > 2 else "spec_free.yaml"
i2t = {int(v): str(k) for k, v in E.token_to_idx.items()}
model = rfd.Proteina.from_pretrained(f"{T}/bundle_189", device="cuda"); model.eval()
orig = model.autoencoder.decode; seen = {"n": 0}
def wrap(batch, *a, **k):
    if seen["n"] < 1:
        seen["n"] += 1; cs = batch[C_SEQ.full_name][0]; rt = batch.get("residue_type"); tt = batch.get("token_type")
        n = cs.shape[0]; print("tokens", n, "keys with seq/ident:", [x for x in batch.keys() if "seq" in x.lower() or "residue" in x.lower() or "token" in x.lower()][:12], flush=True)
        for i in range(n):
            r = i2t.get(int(rt[0][i]), "?") if rt is not None else "-"; c = i2t.get(int(cs[i]), "?"); t = int(tt[0][i]) if tt is not None else -1
            if i >= n - 10: print(f"  token {i}: input residue_type {r}  C_SEQ(at decode) {c}  token_type {t}", flush=True)
    out = orig(batch, *a, **k)
    if seen["n"] == 1:
        seen["n"] = 2; rt2 = out["residue_type"][0]; print("decoder output residue_type at NCAA-type tokens:", [(i, i2t.get(int(rt2[i]), "?")) for i in range(rt2.shape[0]) if batch.get("token_type") is not None and int(batch["token_type"][0][i]) == 6], flush=True)
    return out
model.autoencoder.decode = wrap
orig_nn = model.call_nn; seen2 = {"n": 0}
def wrap_nn(b, *a, **k):
    if seen2["n"] < 1:
        seen2["n"] += 1; cs = b[C_SEQ.full_name][0]; tt = b["token_type"][0]; rt = b["residue_type"][0]
        print("DENOISER INPUT at NCAA-type tokens (token_type 6): ", [(i, "input_res=" + i2t.get(int(rt[i]), "?"), "C_SEQ=" + i2t.get(int(cs[i]), "?")) for i in range(cs.shape[0]) if int(tt[i]) == 6], flush=True)
        print("DENOISER INPUT peptide canonical tokens C_SEQ:", [(i, i2t.get(int(cs[i]), "?")) for i in range(cs.shape[0]) if int(tt[i]) == 0][-8:], flush=True)
    return orig_nn(b, *a, **k)
model.call_nn = wrap_nn
ov = ["~metrics", f"inputs={T}/gen_test/{spec}", f"out_dir={T}/gen_test/runs/probe_identity_{int(__import__("time").time())}", "num_replicates=1", "batch_size=1", "seed=1234", f"subset_to_examples=[{ex}]"]
rfd.generate(model, overrides=ov); print("PROBE_DONE", flush=True)
