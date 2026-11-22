"""Figure: relaxed interface energy test (PyRosetta REF2015): alanine-scan sanity on natural complexes, repair control vs conversion."""
import os, csv, json, glob, statistics as S
import matplotlib.pyplot as plt

N = "C:/Research/RFD4_fine_tune/data/natural_complexes"
ARMS = "C:/Research/Dataset_ncaa/real_benchmark/pilot/ncf2/out/iter3_arms.tsv"


def fig(OUT, C):
    f, ax = plt.subplots(1, 3, figsize=(16.5, 5.8), gridspec_kw=dict(wspace=0.42))
    a = ax[0]
    rows = []
    for p in glob.glob(f"{N}/energy_*.tsv"):
        if os.path.basename(p).startswith(("energy_conv", "energy_rep")):
            continue
        try:
            rows += list(csv.DictReader(open(p), delimiter="\t"))
        except Exception:
            pass
    rows = [r for r in rows if "ddg_ala" in r]
    for r in rows:
        a.scatter(float(r["sc_atoms"]), float(r["ddg_ala"]), color="#2a6f97" if r["sc_contact"] == "1" else "#8d99ae", s=45, edgecolor="#222", zorder=3)
    a.axhline(0, color="#222", lw=0.8)
    a.set_xlabel("side-chain atoms within 4.5 A of the protein"); a.set_ylabel("energy cost of mutating the residue to Ala\n(REU, higher = the residue mattered more)")
    yin = [float(r["ddg_ala"]) for r in rows if r["sc_contact"] == "1"]; yout = [float(r["ddg_ala"]) for r in rows if r["sc_contact"] == "0"]
    a.set_title(f"(a) Alanine scan, 3 natural complexes\n{len(rows)} mutations; touching residues\nmedian {S.median(yin):.1f} REU, others {S.median(yout):.1f}", fontsize=10.5, loc="left", fontweight="bold")
    b = ax[1]
    sm = json.load(open(f"{N}/energy_summary.json"))
    data = [sm["repair"], sm["conv"]]
    bp = b.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
    for p, c in zip(bp["boxes"], ["#2a6f97", "#c9a227"]):
        p.set_facecolor(c); p.set_alpha(0.6)
    for i, d in enumerate(data):
        b.scatter([i + 1 + 0.12 * ((j % 5) - 2) / 2 for j in range(len(d))], d, color="#222", s=14, zorder=3)
        b.text(i + 1, max(d) + 3, f"median {S.median(d):+.1f}\n(n={len(d)})", ha="center", fontsize=9, fontweight="bold")
    b.axhspan(-5, 5, color="#d8f0dc", alpha=0.6, zorder=0); b.text(0.55, -4.5, "+-5 REU: pipeline noise", fontsize=8.5)
    b.set_xticks([1, 2]); b.set_xticklabels(["NCFlow repair\n(same residue types)", "conversion\n(residue types changed)"]); b.set_ylabel("interface energy after minus before (REU)\n(positive = less favourable)")
    b.set_title("(b) Energy change: repair vs\nconversion (shaded = noise)", fontsize=10.5, loc="left", fontweight="bold")
    c = ax[2]
    last = {}
    for r in csv.DictReader(open(ARMS), delimiter="\t"):
        if r["arm"] == "repair" and (r["item"] not in last or int(r["round"]) > int(last[r["item"]]["round"])):
            last[r["item"]] = r
    xs, ys = [], []
    for p in glob.glob(f"{N}/energy_conv_*.json"):
        if "inputs" in os.path.basename(p):
            continue
        e = json.load(open(p))
        if "before_dG" in e and "after_dG" in e and e["before_dG"] < -20 and e["after_dG"] < -20 and e["item"] in last:
            xs.append(float(last[e["item"]]["contact_retention"])); ys.append(e["after_dG"] - e["before_dG"])
    c.scatter(xs, ys, color="#c9a227", s=50, edgecolor="#222")
    c.axvline(0.9, color="#222", ls=":"); c.axhspan(-5, 5, color="#d8f0dc", alpha=0.6, zorder=0)
    c.set_xlabel("residue contacts kept (geometric rule, 0.9 line)"); c.set_ylabel("interface energy change (REU)")
    c.set_title("(c) Geometry and energy are different questions\n(converted peptides)", fontsize=10.5, loc="left", fontweight="bold")
    f.text(0.01, -0.04, "Result: the energy score discriminates interface residues (a) and the repair costs about nothing (b, median +2.3), but converted peptides are typically less favourable (median +14): keeping the contacts is not enough (c).", fontsize=9.5, fontweight="bold")
    f.savefig(os.path.join(OUT, "fig24_energy_test.png"), bbox_inches="tight")
    plt.close(f)
