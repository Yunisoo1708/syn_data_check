import os, sys, re
sys.path.insert(0, os.path.expanduser("~/rfd4_trial/patch")); sys.path.insert(0, os.path.expanduser("~/rfd4_trial/code"))
import _common, rfd4_proteina as rfd
T = os.path.expanduser("~/rfd4_trial")
model = rfd.Proteina.from_pretrained(f"{T}/bundle_189", device="cpu")
for n, p in model.nn.named_parameters():
    if re.search(r"sequence|readout|embed|cond_factory|insertion", n) and "transformer_layers" not in n:
        print(n, tuple(p.shape), p.numel())
print("modules:", [n for n, m in model.nn.named_children()][:30])
