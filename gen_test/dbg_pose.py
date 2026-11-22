import sys; sys.path.insert(0, "/mnt/c/Research/RFD4_fine_tune/gen_test")
import analyze_gen as A, numpy as np, gemmi
npep, nrec, sg = A.native("7XAU_F")
f = "/mnt/c/Research/RFD4_fine_tune/gen_test/out_local/control_v5/inference_samples/7XAU_F_0/7XAU_F_0_generated.cif"
m = gemmi.read_structure(f)[0]; P = [c for c in m if c.name == "P"][0]
grec = [r for c in m if c.name != "P" for r in c]
print("n receptor residues gen", len(grec), "native", len(nrec))
G = np.array([A.xyz(r, "CA") for r in grec]); N = np.array([A.xyz(r, "CA") for r in nrec])
R, t = A.kabsch(G, N); G2 = (R @ G.T).T + t
print("receptor CA rmsd after superposition", float(np.sqrt(((G2 - N) ** 2).sum(-1).mean())))
pc = np.array([a.pos.tolist() for r in P for a in r]); pc = (R @ pc.T).T + t
npc = np.array([a.pos.tolist() for r in npep for a in r])
rx = np.array([a.pos.tolist() for r in nrec for a in r])
print("peptide centroid designed", pc.mean(0).round(1), "native", npc.mean(0).round(1), "receptor centroid", rx.mean(0).round(1))
d = np.linalg.norm(pc[:, None] - rx[None], axis=-1); print("min dist", d.min().round(2), "pairs<2A", int((d < 2).sum()), "pairs<4.5", int((d < 4.5).sum()))
dn = np.linalg.norm(npc[:, None] - rx[None], axis=-1); print("native: min", dn.min().round(2), "pairs<4.5", int((dn < 4.5).sum()), "atoms designed", len(pc), "native", len(npc))
