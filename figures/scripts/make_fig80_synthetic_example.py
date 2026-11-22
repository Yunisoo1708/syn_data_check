"""Fig 80: one synthetic training complex (3RSF, data/bal_pocket12a). Needs fig80_synthetic_example_raw.png from render_fig80_synthetic_example.py (pymol-oss python)."""
import matplotlib.pyplot as plt, matplotlib.image as mpimg
D = "C:/Research/RFD4_fine_tune/figures"
f = plt.figure(figsize=(15, 7.2)); ax = f.add_axes([0.0, 0.03, 0.62, 0.88]); ax.imshow(mpimg.imread(f"{D}/fig80_synthetic_example_raw.png")); ax.axis("off")
ax.annotate("D-Trp (the NCAA)\nswapped in by NCFlow", xy=(0.665, 0.50), xycoords="axes fraction", xytext=(0.72, 0.12), textcoords="axes fraction", fontsize=11, color="#b000b0", fontweight="bold", arrowprops=dict(arrowstyle="->", color="#b000b0", lw=1.6))
ax.text(0.10, 0.70, "natural residues\nof the peptide", transform=ax.transAxes, fontsize=10.5, color="#d2691e", fontweight="bold")
ax.text(0.04, -0.02, "grey: receptor pocket (lines = residues within 5 A of the peptide)", transform=ax.transAxes, fontsize=9, color="#444")
t = f.add_axes([0.63, 0.05, 0.36, 0.86]); t.axis("off")
rows = [("Complex", "3RSF peptide-protein (natural, PDB), pocket cropped to 12 A"),
        ("Peptide (7 res.)", "ALA-ALA-[D-TRP]-LEU-PHE-GLU-ALA"),
        ("What changed", "residue 3: L-Trp  ->  D-Trp (one swap, the rest is natural)"),
        ("Made by", "NCFlow: 4 draws of the swapped residue, mirror-image repair, then 5 gates"),
        ("Gates passed", "strict chemistry; no receptor atom < 2.2 A; contacts kept (96% of the residue contacts, 100% polar); CA drift <= 1.0 A; contact ratio 0.9-1.3"),
        ("Used for", "training the denoiser (LoRA) and the decoder: the real NCAA complexes stay test-only")]
y = 0.97
for k, v in rows:
    t.text(0.0, y, k, fontsize=10.5, fontweight="bold", color="#264653", va="top"); t.text(0.0, y - 0.045, v, fontsize=9.6, va="top", wrap=True); y -= 0.165 if len(v) < 75 else 0.215
f.suptitle("One synthetic training complex", fontsize=14, fontweight="bold", x=0.02, ha="left", y=0.98)
f.savefig(f"{D}/fig80_synthetic_example.png", dpi=130); print("saved")
