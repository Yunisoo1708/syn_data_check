import os, sys
sys.path.insert(0, "/mnt/c/Research/Dataset_ncaa/multigpu/rfd4_ncaa_patch"); sys.path.insert(0, "/mnt/c/Research/RFD4_fine_tune")
os.environ["RFD4_VOCAB_YAML"] = "/mnt/c/Research/Dataset_ncaa/multigpu/rfd4_ncaa_patch/aa_rna_dna_glycan_ncaa_ext_v3.yaml"
import _common, rfd4_proteina as rfd, torch
m = rfd.Proteina.from_pretrained("/mnt/c/Research/Dataset_ncaa/rfd4_bundle_189", device="cpu")
def cnt(x): return sum(p.numel() for p in x.parameters())
print("TOP: Proteina children:", [(k, f"{cnt(v)/1e6:.1f}M") for k, v in m.named_children()])
print("autoencoder children:", [(k, f"{cnt(v)/1e6:.1f}M") for k, v in m.autoencoder.named_children()])
n = m.nn
print("FLOW NETWORK (model.nn) children:", [(k, f"{cnt(v)/1e6:.1f}M") for k, v in n.named_children()])
for k, v in n.named_children():
    sub = [(a, f"{cnt(b)/1e6:.2f}M") for a, b in v.named_children()][:8]
    print(" ", k, "->", sub)
b0 = n.transformer_layers[0]
print("BLOCK 0 children:", [(a, f"{cnt(b)/1e6:.2f}M") for a, b in b0.named_children()])
for a, b in b0.named_children():
    print("   ", a, "->", [(c, f"{cnt(d)/1e6:.2f}M") for c, d in b.named_children()])
for name in ("ca_linear", "local_latents_linear"):
    mod = getattr(n, name, None)
    print(name, mod)
