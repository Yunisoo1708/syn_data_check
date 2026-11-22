"""Generation test: design the peptide around a fixed receptor pocket (real test complexes) with the 189-class flow model (control or + a LoRA adapter, merged) and a chosen decoder; non-canonical residue identities given at their native positions.
usage: gen_run.py NAME ADAPTER_or_none DECODER.pt OUT_DIR [EXAMPLES comma list] [N replicates]"""
import os, sys
T = os.path.expanduser("~/rfd4_trial")
sys.path.insert(0, f"{T}/patch"); sys.path.insert(0, f"{T}/code")
import _common
import torch, peft as pt, rfd4_proteina as rfd
name, adapter, dec, out = sys.argv[1:5]; ex = sys.argv[5] if len(sys.argv) > 5 else ""; nrep = sys.argv[6] if len(sys.argv) > 6 else "4"
model = rfd.Proteina.from_pretrained(f"{T}/bundle_189", device="cuda")
print("decoder:", "BUNDLE (vanilla)" if dec == "bundle" else model.autoencoder.decoder.load_state_dict(torch.load(dec, map_location="cuda"), strict=True), flush=True)
if adapter != "none":
    model.nn = pt.PeftModel.from_pretrained(model.nn, adapter, is_trainable=False); model.nn = model.nn.merge_and_unload()
model.eval()
ov = ["~metrics", f"inputs={T}/gen_test/{os.environ.get('GEN_SPEC', 'spec_gen.yaml')}", f"out_dir={out}", f"num_replicates={nrep}", f"batch_size={nrep}", f"seed={os.environ.get('GEN_SEED', '1234')}"]
if ex: ov.append(f"subset_to_examples=[{ex}]")
print("overrides:", ov, flush=True)
rfd.generate(model, overrides=ov)
print("GENERATE_DONE", name, flush=True)
