"""Render one synthetic training complex (NCFlow-converted, data/bal_pocket12a/cif/3RSF__p1_bal.cif: the natural peptide with one residue swapped for D-Trp). Run with the pymol-oss python."""
from pymol import cmd
cmd.load(r"C:/Research/RFD4_fine_tune/data/bal_pocket12a/cif/3RSF__p1_bal.cif", "cx")
cmd.bg_color("white"); cmd.hide("everything")
cmd.select("pep", "chain P"); cmd.select("rec", "not chain P"); cmd.select("nc", "chain P and resn DTR"); cmd.select("pocket", "byres (rec within 5 of pep)")
cmd.show("cartoon", "rec"); cmd.color("gray70", "rec"); cmd.set("cartoon_transparency", 0.3)
cmd.show("lines", "pocket"); cmd.color("gray40", "pocket and elem C"); cmd.set("line_width", 2.5)
cmd.show("sticks", "pep"); cmd.color("tv_orange", "pep and elem C"); cmd.color("magenta", "nc and elem C")
for e, c in (("N", "blue"), ("O", "red")): cmd.color(c, f"pep and elem {e}")
cmd.set("stick_radius", 0.25); cmd.set("ray_shadows", 0); cmd.set("antialias", 2)
cmd.orient("pep"); cmd.zoom("pep", 7)
cmd.png(r"C:/Research/RFD4_fine_tune/figures/fig80_synthetic_example_raw.png", width=1500, height=1100, dpi=150, ray=1)
