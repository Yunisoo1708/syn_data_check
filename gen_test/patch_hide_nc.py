"""Patch the BOX library file rfproteina/datapipes/pipeline.py (venv) so that the identity of non-canonical (NCAA) tokens can be HIDDEN at generation.
Why (Results.MD 71, 81): the library fills the sequence condition (C_SEQ) and the element feature of NCAA tokens from the input structure even when the spec has no C_SEQ condition for them,
so the model was always told which NCAA to build. With env HIDE_NC_IDENTITY_CHAIN_PREFIX=P the NCAA tokens (token type 6, env HIDE_NC_TOKEN_TYPE) of chains starting with that prefix get C_SEQ = <M>, no element and are treated as NOT sequence-conditioned;
the token type stays (they are still NCAA-type, so the decoder only chooses among NCAA classes). Without the env variable nothing changes. Idempotent; backup pipeline.py.bak_before_hidenc.
usage (on the box): python3 patch_hide_nc.py"""
import os, shutil
import sys
p = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/rfd4_venv/lib/python3.12/site-packages/rfproteina/datapipes/pipeline.py"); s = open(p).read()   # the file the GENERATOR imports (code/pipeline.py is only used by the eval scripts)
if "HIDE_NC_IDENTITY_CHAIN_PREFIX" in s: print("already patched"); raise SystemExit
shutil.copy(p, p + ".bak_before_hidenc")
anchor = '        seq_annotation = C_SEQ.annotation(aa, default="generate")  # [L] res_name strings; MASKED where unset\n'
assert anchor in s
add = anchor + '''        _hnp = os.environ.get("HIDE_NC_IDENTITY_CHAIN_PREFIX")
        if _hnp:  # free-choice test: do NOT give the identity of NCAA tokens of the chains starting with this prefix (token type kept)
            _hide = (token_type == int(os.environ.get("HIDE_NC_TOKEN_TYPE", "6"))) & np.char.startswith(aa.chain_id.astype(str), _hnp) & ~aa.atomize
            seq_cond_mask = seq_cond_mask & ~_hide
            seq_annotation = np.array(seq_annotation, dtype=object); seq_annotation[_hide] = MASKED
'''
s = s.replace(anchor, add); open(p, "w").write(s); print("patched", p)
