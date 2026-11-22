import sys
from atomworks.io.parser import parse
r = parse(sys.argv[1]); aa = r["assemblies"]["1"][0]
import numpy as np
print("n atoms", len(aa), "chains", sorted(set(aa.chain_id)))
for c in sorted(set(aa.chain_id)):
    m = aa.chain_id == c; ids = sorted(set(aa.res_id[m].tolist())); print(c, "res_id", ids[:8], "...", ids[-3:], "n", len(ids), "names", sorted(set(aa.res_name[m]))[:6])
print([a for a in aa.get_annotation_categories()][:20])
