import os, sys
HERE = "/mnt/c/Research/RFD4_fine_tune"
sys.path.insert(0, "/mnt/c/Research/Dataset_ncaa/multigpu/rfd4_ncaa_patch"); sys.path.insert(0, HERE)
os.environ["RFD4_VOCAB_YAML"] = "/mnt/c/Research/Dataset_ncaa/multigpu/rfd4_ncaa_patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml"
import _common, rfd4_proteina as rfd, torch
m = rfd.Proteina.from_pretrained("/mnt/c/Research/Dataset_ncaa/rfd4_bundle_189", device="cpu")
n = m.nn
tot = sum(p.numel() for p in n.parameters()); print("flow network parameters:", tot)
import collections
c = collections.Counter(); sizes = collections.Counter()
for name, mod in n.named_modules():
    if isinstance(mod, torch.nn.Linear):
        key = ".".join(x for x in name.split(".") if not x.isdigit())
        c[key] += 1; sizes[key] += sum(p.numel() for p in mod.parameters())
for k, v in sorted(sizes.items(), key=lambda x: -x[1])[:22]: print(f"{c[k]:4d} x  {k}   total {v/1e6:.2f}M")
