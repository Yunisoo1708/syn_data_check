"""Figure: diversity of the denoiser/decoder training pool (train_all_cif_clean2): which non-canonical residues, how many protein families and entries, peptide lengths, share of non-canonical residues per complex."""
import os, json, re, collections, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D = "C:/Research/RFD4_fine_tune"; POOL = f"{D}/data/train_all_cif_clean2"
CANON = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
DFORMS = set("DSN DGL DTH DAS DLY DGN DSG DIL DHI DCY DTY MED DTR".split())
GEN = set("23F 4FB FP9 HYP 1AC AIB 2ML SAR 04Q MAA MLE MVA".split())
MAN = {"first batch": "pilot_pocket12", "top-up 1": "topup_pocket12", "top-up 2": "topup2_pocket12", "top-up 3": "topup3_pocket12", "general a": "gen_pocket12", "general b": "gen_pocket12b", "general c": "gen_pocket12c", "general d": "gen_pocket12d"}
rec = {}
for lab, d in MAN.items():
    for m in json.load(open(f"{D}/data/{d}/manifest.json")): rec[os.path.basename(m["cif"])] = (lab, m)
G = json.load(open(f"{D}/data/lists/complex_groups.json"))
files = sorted(os.listdir(POOL)); rows = []
for f in files:
    if f not in rec: continue
    lab, m = rec[f]; seq = m["sequence"]; ent = f.split("_")[0].upper()
    n = 0
    with open(f"{POOL}/{f}", errors="ignore") as fh:
        for line in fh:
            if line.startswith("ATOM") and " CA " in line: n += 1
    rows.append(dict(f=f, batch=lab.split()[0] if lab.startswith("general") else ("first batch" if lab == "first batch" else "top-up"), seq=seq, ent=ent, fam=G.get(ent, ent), pep=len(seq), total=n))
print("complexes with a manifest record:", len(rows), "of", len(files))
batch_ct = collections.Counter(r["batch"] for r in rows)
code_ct = collections.Counter(); code_cx = collections.defaultdict(set)
for r in rows:
    for c in r["seq"]:
        if c not in CANON: code_ct[c] += 1; code_cx[c].add(r["f"])
nnc = np.array([sum(c not in CANON for c in r["seq"]) for r in rows]); plen = np.array([r["pep"] for r in rows]); tot = np.array([r["total"] for r in rows])
fam_ct = collections.Counter(r["fam"] for r in rows); ent_ct = collections.Counter(r["ent"] for r in rows)
share = 100 * nnc / np.maximum(tot, 1)
print(f"codes {len(code_ct)}; positions {sum(code_ct.values())}; entries {len(ent_ct)}; families {len(fam_ct)}; median peptide len {np.median(plen):.0f}; noncanonical per complex median {np.median(nnc):.0f}; share of residues non-canonical: median {np.median(share):.1f}% mean {share.mean():.1f}%")
print("D-form positions", sum(v for k, v in code_ct.items() if k in DFORMS), "general-12 positions", sum(v for k, v in code_ct.items() if k in GEN), "other", sum(v for k, v in code_ct.items() if k not in DFORMS | GEN))
print("batches", dict(batch_ct)); print("top families", fam_ct.most_common(5), "complexes per entry max", max(ent_ct.values()), "median", np.median(list(ent_ct.values())))

f = plt.figure(figsize=(19, 10.6)); gs = f.add_gridspec(2, 3, height_ratios=[0.55, 2.3], hspace=0.38, wspace=0.32)
s = f.add_subplot(gs[0, :]); s.axis("off"); s.set_xlim(0, 100); s.set_ylim(0, 10)
box = [("Natural complexes", "real PDB peptide-protein complexes\n(train split only, protein families\nof the real test never used)", "#dde5b6"),
       ("Residue swaps (NCFlow)", "NCFlow rewrites peptide residues as non-canonical ones,\nbackbone generated too, kept within 1 A (CA RMS) of the\nnatural pose; real receptor, contacts >= 90% kept", "#ade8f4"),
       ("Training pool", f"{len(rows):,} complexes from {len(ent_ct):,} PDB entries\n({len(fam_ct)} protein families), checked by\nthe trainer pipeline", "#ffd6a5")]
for i, (t, d, c) in enumerate(box):
    x = 1 + i * 33; s.add_patch(FancyBboxPatch((x, 1), 30, 8, boxstyle="round,pad=0.2", fc=c, ec="#555")); s.text(x + 15, 7.4, t, ha="center", va="center", fontsize=11, fontweight="bold"); s.text(x + 15, 3.7, d, ha="center", va="center", fontsize=8.5)
for x0, x1 in [(31.5, 33.5), (64.5, 66.5)]: s.annotate("", xy=(x1, 5), xytext=(x0, 5), arrowprops=dict(arrowstyle="->", color="#555"))
s.set_title("Method: what the training pool is made of", fontsize=10.5, loc="left", fontweight="bold")

a = f.add_subplot(gs[1, 0]); top = code_ct.most_common(32)[::-1]
col = ["#b23a48" if c in DFORMS else ("#2a6f97" if c in GEN else "#8d99ae") for c, _ in top]
a.barh([c for c, _ in top], [v for _, v in top], color=col, edgecolor="#555"); a.tick_params(axis="y", labelsize=7.5)
for i, (c, v) in enumerate(top): a.text(v + max(v for _, v in top) * 0.01, i, f"{v:,}", va="center", fontsize=7)
a.set_xlabel("non-canonical peptide positions in the pool"); a.set_title(f"(a) Residue types ({len(code_ct)} codes, top 32)", fontsize=10.5, loc="left", fontweight="bold")
from matplotlib.patches import Patch
a.legend(handles=[Patch(color="#b23a48", label="13 D-forms"), Patch(color="#2a6f97", label="12 general non-D codes"), Patch(color="#8d99ae", label="other (kept from the natural entry)")], frameon=False, fontsize=8, loc="lower right")

b = f.add_subplot(gs[1, 1]); names = ["first batch", "top-up", "general"]; vals = [batch_ct[n] for n in names]
b.bar(names, vals, color=["#c9a227", "#b23a48", "#2a6f97"], edgecolor="#555")
for i, v in enumerate(vals): b.text(i, v + 40, f"{v:,}", ha="center", fontsize=9)
b.set_ylabel("complexes"); b.set_title("(b) Source of the complexes", fontsize=10.5, loc="left", fontweight="bold")
ins = b.inset_axes([0.08, 0.52, 0.40, 0.4]); ec = np.array(sorted(ent_ct.values())); ins.hist(ec, bins=np.arange(1, ec.max() + 2) - 0.5, color="#8d99ae", edgecolor="#555")
ins.set_xlabel("complexes per PDB entry", fontsize=7.5); ins.set_ylabel("entries", fontsize=7.5); ins.tick_params(labelsize=7); ins.set_title("same receptor reused", fontsize=8)

c = f.add_subplot(gs[1, 2]); c.hist(plen, bins=np.arange(0, 60, 2), color="#2a6f97", edgecolor="#555", alpha=0.85, label="peptide length")
c.set_xlabel("peptide length (residues)"); c.set_ylabel("complexes")
c2 = c.twinx(); c2.hist(share, bins=np.arange(0, 40, 1), histtype="step", color="#b23a48", lw=2, label="non-canonical share of all residues")
c2.set_ylabel("complexes (red)"); c2.tick_params(axis="y", colors="#b23a48")
c.set_title(f"(c) Peptide length and non-canonical share (median {np.median(share):.1f}% of residues)", fontsize=10.5, loc="left", fontweight="bold")
f.text(0.01, 0.005, f"{len(rows):,} training-pool complexes, {len(ent_ct):,} PDB entries, {len(fam_ct)} protein families (family = sequence-similarity group of the frozen split v4). Non-canonical share = peptide non-canonical residues / all residues of the cropped complex (receptor pocket + peptide).", fontsize=8.5, fontweight="bold")
f.savefig(f"{D}/figures/fig40_training_data_diversity.png", bbox_inches="tight", dpi=130); plt.close(f)
