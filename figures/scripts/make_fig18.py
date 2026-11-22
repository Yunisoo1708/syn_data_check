"""Figure: per-residue map of the 30 natural peptides (side-chain contact, backbone contact, where an N-methyl or alpha-methyl fits). Reads natural_complexes/analysis_*.tsv."""
import os, csv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

D = "C:/Research/RFD4_fine_tune/data/natural_complexes"
ONE = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")


def fig(OUT, C):
    res = list(csv.DictReader(open(f"{D}/analysis_residues.tsv", encoding="utf-8"), delimiter="\t"))
    meta = {r["pdb"]: r for r in csv.DictReader(open(f"{D}/natural_complexes.tsv", encoding="utf-8"), delimiter="\t")}
    cx = {r["pdb"]: r for r in csv.DictReader(open(f"{D}/analysis_complexes.tsv", encoding="utf-8"), delimiter="\t")}
    order = sorted(cx, key=lambda p: (-int(cx[p]["helix"]) / int(cx[p]["n_res"]) + 0.5 * int(cx[p]["ppii"]) / int(cx[p]["n_res"]) * 0 , p))
    # group: helical (>= 40 % helix), polyproline (>= 40 % PPII), rest
    def grp(p):
        n = int(cx[p]["n_res"])
        if int(cx[p]["helix"]) >= 0.4 * n:
            return 0
        if int(cx[p]["ppii"]) >= 0.4 * n:
            return 1
        return 2
    order = sorted(cx, key=lambda p: (grp(p), -int(cx[p]["n_res"]), p))
    gname = {0: "mostly helix", 1: "mostly polyproline II", 2: "extended, turns, mixed"}
    f, ax = plt.subplots(figsize=(15, 12.5))
    col = {(0, 0): "#e9ecef", (1, 0): "#7fb3d5", (0, 1): "#f0b27a", (1, 1): "#a569bd"}
    y = 0; yt, yl = [], []; last = None
    for p in order:
        if grp(p) != last:
            if last is not None:
                y += 0.6
            ax.text(-0.9, y - 0.2, gname[grp(p)], fontsize=10, fontweight="bold", ha="right", color="#333")
            last = grp(p)
        rs = [r for r in res if r["pdb"] == p]
        for r in rs:
            i = int(r["pos"]) - 1
            sc = int(r["sc_contact"]); bb = int(int(r["nh_to_protein"]) > 0 or int(r["co_to_protein"]) > 0)
            ax.add_patch(Rectangle((i, y), 0.95, 0.9, fc=col[(sc, bb)], ec="white", lw=0.8))
            ax.text(i + 0.47, y + 0.5, ONE.get(r["resname"], "X"), ha="center", va="center", fontsize=8.5, color="#111")
            if r["n_methyl"] == "yes":
                ax.add_patch(Rectangle((i + 0.04, y + 0.04), 0.3, 0.14, fc="#1e8449", ec="none"))
            if r["a_methyl"] == "yes":
                ax.add_patch(Rectangle((i + 0.6, y + 0.04), 0.3, 0.14, fc="#c0392b", ec="none"))
        yt.append(y + 0.45)
        yl.append(f"{p}  {meta[p]['prot_desc'][:30]}")
        y += 1
    ax.set_xlim(-0.2, 20.2); ax.set_ylim(y + 0.3, -1.1)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8.3)
    ax.set_xticks([i + 0.45 for i in range(20)]); ax.set_xticklabels([str(i + 1) for i in range(20)], fontsize=8)
    ax.set_xlabel("peptide residue (only the residues seen in the crystal)"); ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    leg = [Patch(fc=col[(1, 0)], label="side chain touches the protein (within 4.5 A)"), Patch(fc=col[(0, 1)], label="backbone N or O within 3.5 A of the protein (hydrogen-bond candidate)"),
           Patch(fc=col[(1, 1)], label="both"), Patch(fc=col[(0, 0)], label="neither"),
           Patch(fc="#1e8449", label="green bar: an N-methyl fits and the NH is free (conservative)"), Patch(fc="#c0392b", label="red bar: an alpha-methyl fits (2.8 A clearance)")]
    ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=8.8)
    ax.set_title("Natural peptides on their proteins: which positions are interface, and where could the backbone be changed?\n(30 complexes with crystal coordinates; 2BB4 left out: only 4 of 9 residues are visible)", fontsize=12, fontweight="bold", loc="left", pad=40)
    f.savefig(os.path.join(OUT, "fig18_natural_peptide_map.png"), bbox_inches="tight")
    plt.close(f)
