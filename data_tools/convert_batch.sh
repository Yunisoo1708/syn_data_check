#!/bin/bash
# usage: convert_batch.sh PKL_DIR OUT_DIR SUFFIX  -> OUT_DIR/{cif,pep_cif,masked/cif,validate_bad.json}; masked+validated complexes are in OUT_DIR/final/
set -e
IN=$1; OUT=$2; SUF=$3
S=/mnt/c/Research/RFD4_fine_tune/splits; R=/mnt/c/Research/RFD4_fine_tune; P=/mnt/c/Research/Dataset_ncaa/multigpu/rfd4_ncaa_patch
AF3=/home/yunisoo/miniconda3/envs/af3/bin/python
sed "s/_ncf\.cif/_$SUF.cif/g" $S/make_pep_cifs.py > $OUT.mk_pep.py; sed "s/_ncf\.cif/_$SUF.cif/g" $S/mask_pilot_cifs.py > $OUT.mk_mask.py
cp $OUT.mk_pep.py $S/_tmp_mk_pep_$SUF.py; cp $OUT.mk_mask.py $S/_tmp_mk_mask_$SUF.py
cd $S && POCKET=12 SUFFIX=$SUF GUARD=v4 $AF3 write_pilot_cifs.py $OUT $IN/*.pkl > $OUT.write.log 2>&1; tail -n 1 $OUT.write.log | cut -c1-300
cd $OUT && $AF3 -c "import json;M=json.load(open('manifest.json'));json.dump({'usable':[m['entry'] for m in M]},open('usable.json','w'));print(len(M),'entries')"
$AF3 $S/_tmp_mk_pep_$SUF.py cif pep_cif
cd $P && $AF3 geo_check.py $OUT/verdicts_pep.tsv "$OUT/pep_cif/*.cif" | tail -n 1
cd $OUT && $AF3 $S/_tmp_mk_mask_$SUF.py cif masked usable.json manifest.json verdicts_pep.tsv $P/aa_rna_dna_glycan_ncaa_ext_v3.yaml | tail -n 1
D=$OUT; export RFD4_VOCAB_YAML=$P/aa_rna_dna_glycan_ncaa_ext_v3.yaml HF_HUB_OFFLINE=1 PATCH_DIR=$P CIF_DIR=$D/masked/cif
env OUT=$D/validate_bad.json PYTHONPATH=/home/yunisoo/binder/envs/rfd4_env/lib/python3.12/site-packages:$P:$R /home/yunisoo/binder/envs/rfd3_env/bin/python $R/validate_train_cifs.py 2>&1 | grep -E "^done"
mkdir -p $OUT/final
$AF3 - <<PY
import json,os,shutil
bad={x['file'] for x in json.load(open('$OUT/validate_bad.json'))}
n=0
for f in os.listdir('$OUT/masked/cif'):
    if f not in bad: shutil.copy('$OUT/masked/cif/'+f,'$OUT/final/'+f); n+=1
print('final kept',n,'dropped',len(bad))
PY
echo CONVERT_DONE
