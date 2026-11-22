load C:/Research/RFD4_fine_tune/train_all_cif_clean/3IVQ_tu.cif, cx
hide everything
bg_color white
set ray_opaque_background, on
set antialias, 2
set depth_cue, 0
select pep, cx and chain P
select rec, cx and not chain P
select pocket, byres (rec within 4.5 of pep) and not name N+C+O
select dforms, pep and (resn DAS+DHI)
show cartoon, rec
color gray85, rec
set cartoon_transparency, 0.6, rec
show lines, pocket
color gray50, pocket
set line_width, 2
show sticks, pep
set stick_radius, 0.22, pep
color tv_orange, pep and elem C
set stick_radius, 0.34, dforms
color tv_green, dforms and elem C
util.cnc pep
set label_size, 26
set label_color, black
set label_outline_color, white
set label_font_id, 7
label dforms and name CA, resn + " " + resi
set label_position, (0,3.2,3)
orient pep
zoom pep, 3
set field_of_view, 20
png C:/Research/RFD4_fine_tune/figures/pymol_example_3IVQ.png, 1500, 1000, dpi=200, ray=1
quit
