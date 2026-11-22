load C:/Research/RFD4_fine_tune/data/bal_pocket12a/cif/3RSF__p1_bal.cif, cx
bg_color white
hide everything
select pep, chain P
select rec, not chain P
select nc, chain P and resn DTR
select pocket, byres (rec within 5 of pep)
show cartoon, rec
color gray70, rec
show lines, pocket
color gray40, pocket and elem C
show sticks, pep
color tv_orange, pep and elem C
color magenta, nc and elem C
set stick_radius, 0.25
orient pep
zoom pep, 7
