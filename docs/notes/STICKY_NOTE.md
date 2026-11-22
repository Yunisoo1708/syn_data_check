# RFD4 structure: sticky note

## DATA FLOW MAP of my fine_tuning.py  (UPDATE EVERY STEP; shapes verified by running unless marked ?)

B = structures per batch (3), L = tokens. L differs per structure (53, 52, 57), L_max = longest in batch (57).

```
STAGE 0   files                list[Path]      [L07-0000189.cif, L07-0000689.cif, ...]   31,911 in all_structures
   │      DesignProblemDataset(chunk_of_files, transform=pipe)      chunks <= 2000 files (else quadratic slowdown)
   │      ConcatDataset(chunks) -> one ds, len(ds) = number of files
   ▼
STAGE 1   ds[i] = ONE structure        plain dict      (this is where "feats" lives)
   │      {
   │        "example_id" : str                      "L07-0000189"
   │        "atom_array" : AtomArray, "extra_info", "task", "specification"
   │        "feats"      : dict  <--- THE TENSORS ARE INSIDE THIS DICT, one structure, L = 53 here
   │             "coords"                     [L,37,3]    float32
   │             "atom37_mask"                [L,37]      bool
   │             "token_type", "residue_type" [L]         int
   │             "element", "charge"          [L,37]      int
   │             "incident_bond_type_counts"  [L,37,10]   int16
   │             ... ~50 tensors in all (+ 28 condition_sequence-probability-* [L])
   │      }
   │
   │      DataLoader(ds, batch_size=3, collate_fn=partial(collate_batch, schema=SCHEMA))
   │      does:  [ds[0], ds[1], ds[2]]  ->  collate_batch(that list)
   │               1. from_dict OPENS "feats" and lifts every tensor out   (schema_proteina.py:135)
   │               2. pads each tensor along L up to L_max (pad value from SPECS, schema_proteina.py:58)
   │               3. stacks the 3 structures on a NEW first dim B
   ▼
STAGE 2   training_batch               PaddedBatchedDataSample   (dict-like: keys(), items(), b["x"], b["x"]=v)
   │      there is NO "feats" key any more: its tensors are now TOP-LEVEL keys
   │        batch["coords"]       [B,L_max,37,3]   = [3,57,37,3]   float32
   │        batch["atom37_mask"]  [3,57,37]        bool
   │        batch["token_type"]   [3,57]           int32
   │        batch["padding_mask"] [3,57]           bool   True = real token, False = pad  (new in collate)
   │        ... 78 tensors, all on CPU
   │      non-tensors: example_id/atom_array/task = list of B items; extra_info/specification = dict
   │      "tensors" key = a dict holding the same tensors again (dataclass field); ignore it
   │
   │      MY STEP (Stage 2 -> 3):  copy into a plain dict, .to("cuda") only the torch tensors,
   │                               add coords_nm = coords / 10   [3,57,37,3]  (Angstrom -> nm)
   │                               add coord_mask = atom37_mask  [3,57,37]
   ▼
STAGE 3   d = plain dict               tensors on cuda:0  (+ coords_nm, coord_mask)
   │      model.autoencoder.encode(d)         inside torch.no_grad(); encoder is frozen
   │        encoder: 1587 -> 768 -> 12 layers -> 32   (rfproteina_encoder.py:111)
   ▼
STAGE 4   encoder output               dict      (verified)
   │        z_latent   [3,57,32] float32 cuda   <- the clean latent targets for the denoiser
   │        mean       [3,57,32]   log_scale [3,57,32]   z_latent_pre_ln [3,57,32]
   │        emit_mask  [3,57] bool
   ▼
NEXT (Step 3 loss):  add_clean_samples -> corrupt_batch -> call_nn -> compute_loss
                     (rfproteina_proteina.py:740 / flow_matcher:222 / proteina:524 / flow_matcher:307)
```

Why no "feats" after the DataLoader: collate opens each structure's "feats" dict and stacks its
tensors by name, so `ds[i]["feats"]["coords"]` [L,37,3] becomes `batch["coords"]` [B,L,37,3].

### What is inside `feats`  (77 tensors; checked on L07-0000189.cif with build_design_train_pipeline, L = 53)

`L` = tokens of this structure. Every feats tensor has L as its first dim, except one `[3]`.
"ENC" = read by the encoder's 1587-dim input (from the table further down). Meanings marked (?) are inferred from the name.

```
GROUP                  key                                shape          meaning / what I saw
---------------------  ---------------------------------  -------------  ------------------------------------------------
IDENTITY (9)           tok_idx                            [L]            0..L-1, position of the token
                       residue_type                       [L]            vocab id 0..146 (this file: all 32)
                       condition_sequence_1_residue       [L]            copy of residue type used as a condition (?)
                       token_type  (ENC)                  [L]            aa0 rna1 dna2 atomized3 unknown4 glycan5 ncaa6 (this file: all 3)
                       chain_iid  (ENC, ?)                [L]            chain id (all 0 = one chain)
                       molecule_iid / molecule_entity_id  [L]            molecule ids (all 0)
                       insertion_source_idx               [L]            -1 here (?: bookkeeping for inserted tokens)
                       loss_weight                        [L]            1.0 per token = weight in the loss

FLAGS (4, bool)        supports_local_latent              [L]            token gets a latent z (this file: 0/53 True)
                       is_resolved_mask                   [L]            coordinates observed (53/53 True)
                       is_in_context_conditioning         [L]            token is given as fixed context (0/53)
                       coord_cond_mask                    [L]            coordinates are given as a condition (0/53)

GEOMETRY (5)           coords  (ENC)                      [L,37,3]       xyz in Angstrom, centered near 0 (range -6.6..6.6)
                       atom37_mask  (ENC)                 [L,37]         which of the 37 atom slots are real (this file: 53 of 1961)
                       element  (ENC)                     [L,37]         element id per slot, -1 = empty slot
                       ground_truth_element               [L,37]         element id, 0 = empty
                       charge  (ENC)                      [L,37]         formal charge id (all 0)

BONDS / CHEMISTRY (10) incident_bond_type_counts (ENC)    [L,37,10]      count of each bond type touching each atom slot
                       bond_mask                          [L,L]          True where two tokens are bonded (116/2809 = 58 bonds)
                       bond_order                         [L,L]          0/1/2 (single/double, ?)
                       is_aromatic                        [L,L]          aromatic bond flag
                       token_bond_path_length             [L,L]          shortest bond path between tokens (0..22)
                       canonical_polymer_ordering         [L,L]          all 0 here (?)
                       chirality                          [L,18]         chirality descriptors (all 0 here)
                       chiral_frame_atoms                 [L,6,8]        -1 = none here
                       chiral_slot_owner                  [L,6]          -1 = none here
                       tetrahedral_vertices               [L,4]          -1 = none here

CONDITIONS (44)        28 x condition_sequence-probability-<AA>_1_residue   [L]     per-residue amino-acid probability hint (all 0 here)
                       condition_secondary-structure_1_residue              [L]     helix/strand hint (0)
                       condition_index-group_1_token                        [L]     index group (0)
                       condition_cath-label_1_residue                       [L,3]   fold label, -1 = none
                       condition_ChainCentroid_1_atom                       [L,3]   NaN = none
                       condition_coordinate_1_atom / InternalCoordinate_1_atom  [L,37,3]  NaN = none  (float64!)
                       condition_index_1_residue / CoordinateGroup_1_atom   [L,37]  slot index / group (True / -1)
                       condition_hotspot / rasa-binned / hbond / pistack /
                       pication / saltbridge / hydrophobic / metal-coord    [L,37]  interaction hints (all 0/False here)
                       -> these are optional design-time hints; in my data they are empty defaults

NUCLEIC ACID (4)       na_ss_pair [L,L], na_ss_pair_mask [L,L], na_ss_state [L], na_ss_state_mask [L]   all 0 (unused for peptides)

OTHER (1)              centroid_of_atoms_without_coordinate_conditioning   [3]   ~0 (structure is centered at the origin, ?)
```

### IMPORTANT finding for my NCAA goal (one file only: L07-0000189)

```
token_type  = 3 (atomized) for ALL 53 tokens        residue_type = 32 for all
atom37_mask = exactly 1 real slot per token         supports_local_latent = False for all
=> this molecule was ATOMIZED: every atom is its own token (L = 53 atoms), no residue latent z.
```
So the encoder still runs, but this structure is not a "residue with a latent". Check how many of my
31,911 files are atomized vs real residue/NCAA tokens before training (not checked yet).

```
nn
├── init_repr_factory      seq features (21 feat types: xt_bb_ca, xt_local_latents, token_type,
│                          hotspot, hbond, pistack, saltbridge, ...) -> Linear 897 -> 768
├── cond_factory           time embeddings, cath_code, recycle -> 256-dim conditioning
├── transition_c_1 / _2    conditioning MLPs (SwiGLU)
├── pair_repr_builder      pair features (CA distances, bonds, ...) -> 384-dim pair repr, AdaLN
│
├── transformer_layers[0..13]            14 x  (token_dim 768, 12 heads)
│   ├── mhba                             multi-head attention block
│   │   ├── adaln                        norm_cond, to_gamma, to_beta (conditioned on 256-dim c)
│   │   ├── mha
│   │   │   ├── node_norm
│   │   │   ├── to_qkv        768 -> 2304   <- your target
│   │   │   ├── q/k_layer_norm            (QK-norm)
│   │   │   ├── pair_norm + to_bias       384 -> 12 (pair bias per head)
│   │   │   ├── to_g                      768 -> 768 (gate)
│   │   │   └── to_out_node   768 -> 768    <- your target
│   │   └── scale_output     adaLN-zero gamma
│   └── transition                       SwiGLU MLP
│       ├── adaln
│       ├── transition.swish_linear      768 -> 6144 (SwiGLU)
│       ├── transition.linear_out        3072 -> 768
│       └── scale_output
│
├── pair_update_layers[0..6]             7 layers (every 2nd layer updates the pair repr)
│   ├── linear_x, layer_norm_in
│   ├── tri_mult_out / tri_mult_in       triangle multiplication (cuequivariance)
│   └── transition_out
│
├── s_cycle_norm/proj, z_cycle_norm/proj recycling (up to 9 recycles in training)
├── insertion_q_proj / insertion_k_proj  768 -> 768
├── ca_linear                            768 -> 3   (CA velocity head)
├── local_latents_linear                 768 -> 32  (latent velocity head)
└── sequence_readout                     residue-type head (147 classes, detached)


autoencoder
├── encoder
│   ├── init_repr_factory      atom37 coords, bonds, element/charge -> 768 (Linear 1587 -> 768)
│   ├── transition_c_1 / _2    conditioning MLPs (dim_cond 128)
│   ├── transformer_layers[0..11]    same block layout as nn (mhba + transition),
│   │                                no pair update and no pair bias
│   └── latent_decoder_mean_n_log_scale   -> mean and log-scale of z (32-dim)
└── decoder
    ├── init_repr_factory      ca_coors, z_latent, token_type, residue type (Linear 214 -> 768)
    ├── transformer_layers[0..11]
    ├── struct_linear          768 -> 111 (37 atoms x 3)
    ├── atom_presence_linear   768 -> 37
    └── logit_linear           768 -> 147 (residue type)
```

## Autoencoder data flow (encode -> latent -> decode)

```
INPUT: all-atom residue  (atom37 coords, bonds, element/charge)
│
├── ENCODER
│   ├── init_repr_factory        atom37 + bonds + element/charge -> Linear 1587 -> 768
│   ├── transition_c_1 / _2      conditioning (dim_cond 128)
│   ├── transformer_layers x12   mhba + transition (no pair update, no pair bias)
│   └── latent_decoder_mean_n_log_scale
│       ├── mean       -> 32-dim
│       └── log-scale  -> 32-dim
│
├── SAMPLE z ~ N(mean, exp(log-scale))     per-residue latent, 32-dim
│   (CA coords stay explicit; z carries the side-chain / residue detail)
│
├── DECODER
│   ├── init_repr_factory        ca_coors + z_latent + token_type + residue type -> Linear 214 -> 768
│   ├── transformer_layers x12   mhba + transition
│   ├── struct_linear            768 -> 111  => 37 atoms x 3 coords
│   ├── atom_presence_linear     768 -> 37   => which atoms exist
│   └── logit_linear             768 -> 147  => residue type
│
└── OUTPUT: atom37 coords + atom presence + residue-type logits


HOW nn USES IT (diffusion in latent space)
│
├── nn input:   xt_bb_ca (noisy CA) + xt_local_latents (noisy z, 32-dim)
├── nn output:  ca_linear -> CA velocity (3),  local_latents_linear -> z velocity (32)
├── sampling:   integrate velocities from noise -> clean CA + z
└── final:      DECODER(CA, z) -> full all-atom structure + sequence
```

## How LoRA (PEFT) works

```
y = W·x + (alpha/r) · B·(A·x)
    └ frozen ┘   └ trainable ┘
A: r × in    (random init)
B: out × r   (zeros -> update starts at 0, model begins = pretrained)

to_qkv (768 -> 2304), r=8:  full = ~1.77M params  |  LoRA = ~24.6K (~1.4%)
```

```
get_peft_model / inject_adapter_in_model
├── 1. MATCH    walk named_modules(), pick names ending in target_modules
│                 -> to_qkv, to_out_node in all 14 nn layers = 28 wrapped Linears
├── 2. WRAP     nn.Linear -> lora.Linear
│                 ├── base_layer   (original weights)
│                 ├── lora_A       r × in
│                 └── lora_B       out × r
├── 3. FREEZE   every param without "lora_" in its name -> requires_grad=False
└── 4. FORWARD  base_layer(x) + lora_B(lora_A(dropout(x))) * (alpha/r)
```

```
LoraConfig knobs
├── r                 rank = capacity (8-32 typical)
├── lora_alpha        scaling = alpha/r (keep ratio fixed if r changes)
├── lora_dropout      dropout on the LoRA branch input
├── target_modules    ["to_qkv", "to_out_node"]  (add swish_linear, linear_out to adapt MLPs)
├── modules_to_save   layers trained fully (e.g. ca_linear, local_latents_linear)
└── bias              "none"

After training
├── save_pretrained(dir)    adapter only (few MB)
├── merge_and_unload()      fold B·A into W, no inference cost
└── adapters swappable without touching base weights
```

Setup sketch
```python
cfg = pt.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                    target_modules=["to_qkv", "to_out_node"], bias="none")
model.nn = pt.get_peft_model(model.nn, cfg)   # or pt.inject_adapter_in_model(cfg, model.nn)
model.nn.print_trainable_parameters()
```

Gotchas for fine_tuning.py
- `transformer_layer` -> `transformer_layers`; `str(module)` is not a path (PEFT matches by name)
- keep autoencoder frozen + eval() after `model.train()`
- optimizer: only `p.requires_grad` params

## LoRA variables (every symbol inside the unit)

```
lora.Linear  (one per target: to_qkv, to_out_node)
│
├── INPUT / OUTPUT
│   ├── x                  input tokens            [batch, N_res, in]      in = 768
│   └── y                  output tokens           [batch, N_res, out]     out = 2304 (to_qkv) | 768 (to_out_node)
│
├── FROZEN
│   ├── base_layer         original nn.Linear
│   │   ├── W  (weight)    [out, in]               to_qkv: 2304×768 = 1,769,472
│   │   │                                          to_out_node: 768×768 = 589,824
│   │   └── b  (bias)      [out]                   frozen (bias="none")
│
├── TRAINABLE (requires_grad=True)
│   ├── lora_A.default.weight    A   [r, in]       init: Kaiming uniform (random)
│   └── lora_B.default.weight    B   [out, r]      init: zeros  -> B·A = 0 at step 0
│
├── HYPERPARAMETERS (LoraConfig)
│   ├── r                  rank of the update      e.g. 8
│   ├── lora_alpha         scaling numerator       e.g. 16
│   ├── scaling            alpha / r               e.g. 16/8 = 2.0   (stored as self.scaling)
│   ├── lora_dropout       p applied to x on the LoRA branch only    e.g. 0.05
│   ├── target_modules     ["to_qkv", "to_out_node"]
│   ├── modules_to_save    layers trained fully (optional)
│   ├── bias               "none" | "all" | "lora_only"
│   ├── use_rslora         if True: scaling = alpha / sqrt(r)
│   └── init_lora_weights  True (default: A random, B zero)
│
└── FORWARD
    ├── h  = dropout(x)                     [batch, N_res, in]
    ├── a  = h · Aᵀ                         [batch, N_res, r]      down-project 768 -> r
    ├── d  = a · Bᵀ                         [batch, N_res, out]    up-project r -> out
    ├── y0 = x · Wᵀ + b                     frozen path
    └── y  = y0 + scaling · d               final output
```

```
Parameter count per layer (r = 8)
├── to_qkv        A: 8×768 = 6,144   B: 2304×8 = 18,432   total 24,576
├── to_out_node   A: 8×768 = 6,144   B:  768×8 =  6,144   total 12,288
├── per nn layer  24,576 + 12,288 = 36,864
└── all 14 layers 36,864 × 14 = 516,096 trainable  (vs ~33M in those weights if fully trained)

Named parameters you will see
└── nn.transformer_layers.{0..13}.mhba.mha.to_qkv.lora_A.default.weight
    nn.transformer_layers.{0..13}.mhba.mha.to_qkv.lora_B.default.weight
    nn.transformer_layers.{0..13}.mhba.mha.to_qkv.base_layer.weight   (frozen)
    (same pattern for to_out_node)
```

## Encoder input: where the 1587 comes from (autoencoder.encoder.init_repr_factory)

Rule: every feature is flattened to ONE vector per token ([L, something]), then
concatenated along the last dim. The 37-slot dim is never concatenated across.

```
feature                      raw shape        -> per-token vector     dims
---------------------------  ---------------  ----------------------  ----
token_type                   [L]              embed 7 -> 32            32
x1_a37coors_nm  (+ mask)     [L,37,3]+[L,37]  flatten 111 + 37        148
x1_a37coors_nm_rel (+ mask)  [L,37,3]+[L,37]  flatten 111 + 37        148
chain_idx_seq                [L]              as is                     1
incident_bond_type_counts    [L,37,10]        flatten 37x10           370
element + charge             [L,37] each      embed 119->8, 11->16;
                                              concat -> [L,37,24];
                                              flatten 37x24           888
                                                                     ----
                                                                     1587  ->  Linear 1587 -> 768
```

"Embed" = lookup table. `embedding.weight` is [n_types, dim]; an integer id picks one row.
```
token_type [0, 6, 0]  ->  row 0, row 6, row 0  ->  [3, 32]     (7 types x 32 learned numbers)
element 6 (carbon)    ->  row 6 of [119, 8]    ->  8 numbers
charge 0              ->  row of [11, 16]      -> 16 numbers
per slot: 8 + 16 = 24;  37 slots x 24 = 888
```
Empty slots still get a vector; atom37_mask tells the net which slots are real.

Unverified guesses: charge offset (+5), nm scaling (/10), mask inside the 148, empty element -1 -> row 0.

Where the arrays come from (pipeline.py)
```
ConvertToAtomLevelProteinFeatures  token_type, chain_iid, residue_type, element, charge
ConvertToAtom37Format               coords [L,37,3], atom37_mask, incident_bond_type_counts
build_design_train_pipeline(...)(data) -> {example_id, num_tokens, task, specification,
                                           metrics_tags, extra_info, atom_array, feats}
```

## Decoder input: where the 214 comes from

```
decoder feats_seq (per residue)               dim
├── ca_coors_nm                                 3
├── z_latent_seq                               32
├── token_type                                 32   (learned embedding)
└── cond_residue_type                         147   (one-hot over the 147-token vocabulary)
                                             ───
                                             214  ->  Linear 214 -> 768
```

## Vocabulary (147 tokens) and token_type ids  [rfproteina/encoding.py]

```
147 residue tokens
├── <M> mask (1)  UNK (1)  <A> <G> (atomized placeholders)
├── aa       21   20 canonical + UNK
├── rna       5   A C G U N
├── dna       5   DA DC DG DT DN
├── ncaa    100   SEP TPO CSO PTR CME KCX LLP HYP MLY AIB NLE ORN ... (fixed list, PDB CCD codes)
└── glycan   13   NAG BMA MAN FUC GLC GAL ...

token_type ids: aa 0 | rna 1 | dna 2 | atomized 3 | unknown 4 | glycan 5 | ncaa 6
```

## Do my non-canonical structures fit this encoding?

```
residue                                         what the pipeline does
├── in the 100 NCAA list (CCD code matches)     own token, own residue_type, 37-slot layout -> encoded normally
├── NOT in the list (custom NCAA, beta-AA...)   ATOMIZED: each atom becomes its own token
│                                                 (token_type 3, element/charge features, no local latent z)
└── atom count > 37 or atom names not in the    cannot use the residue layout -> must be atomized
    yaml layout
```

## Where NCAA information is used inside the model  [read from code, not run; encoder feature list not traced]

```
#  what                              where                                         what it does for an NCAA
-  --------------------------------  --------------------------------------------  ------------------------------------------------
1  token_type embedding (id 6=NCAA)  feature_factory.py:462-507 (TokenTypeSeqFeat) Embedding(7,32); row 6 = "this is an NCAA/PTM"
                                     registered feature_factory.py:1442            feeds encoder 1587, decoder 214, nn seq feats
2  residue-type one-hot (WHICH one)  feature_factory.py:400-430                    Embedding(147,147) frozen = identity; NCAA ids are
                                     (ConditionalResidueTypeSeqFeat)               columns; position is <M> if sequence not conditioned
3  output heads (predict which)      decoder logit_linear 768->147                 seq loss autoencoder.py:517-534
                                     nn sequence_readout 147 classes               seq loss flow_matcher.py:410-425
                                     get_allowed_residue_type_mask                 proteina.py:2059-2066: token_type 6 can only
                                                                                   decode to an NCAA token
4  atom layout per residue           decoder.residue_atom_mask[residue_type]       autoencoder.py:159, :576: which atom37 slots this
                                     (table built from rfproteina.encoding)        residue may have = the NCAA chemistry
5  loss weighting by group           VALIDATION_RESIDUE_GROUPS["ncaa"]=NCAA_NAMES  autoencoder.py:30-60 blended_token_weights:
                                                                                   equalises aa/ncaa/glycan/dna/rna (balanced_fraction 0.25)
6  atomized path (not in vocab)      atom_charge / atomic_element features         feature_factory.py:952-1050, x is_atomized
                                     bond_mask / bond_order / is_aromatic          NO residue id, NO latent z (my L07-0000189 case)
```

Takeaways
- Only #1 is truly NCAA-specific; #2-#5 are generic vocabulary machinery that NCAA ids pass through.
- NCAA-aware weights in the pretrained model: token_type row 6, logit_linear + sequence_readout rows for the 100 NCAA ids,
  the 100 one-hot columns. LoRA on to_qkv / to_out_node CANNOT change these; only modules_to_save or a vocab change can.
- A token missing from NCAA_NAMES falls in the "other" loss group (#5).

## The loss dict and how it becomes ONE number  (rfproteina_flow_matcher.py:307 compute_loss, :326 optimization_terms)

`compute_loss(batch, nn_out)` does NOT return a number. It returns a dict: name -> tensor [b] (one loss per structure, before averaging).

```
losses = {
    "bb_ca"                : tensor([3405.0, 812.0, 1290.0])   FM loss, CA positions   (one value per structure)
    "local_latents"        : tensor([  95.0,  40.0,   60.0])   FM loss, 32-d latents
    "insertion_prediction" : tensor([...])    only if present
    "sequence_prediction"  : tensor([...])    only if present
    "signed_volume"        : tensor([...])    only if signed_volume_weight > 0
}                                             (numbers above are made-up examples, shape [3] = batch of 3)
```

Compression (same recipe as training_step, rfproteina_proteina.py ~line 672):

```
STEP A  optimization_terms(losses)      keep only the entries that are trained on
        -> bb_ca, local_latents (always: every data mode) + the optional ones IF present
        (diagnostic-only keys are dropped)

STEP B  .mean() each entry over the batch  ([b] -> one number each)
        bb_ca          (3405 + 812 + 1290) / 3 = 1835.67
        local_latents  ( 95 +  40 +   60) / 3 =   65.0

STEP C  sum the entries                 (plain sum, no extra weights between modes)
        train_loss = 1835.67 + 65.0 = 1900.67      <- shape [] (a scalar)

STEP D  train_loss.backward()  ->  optimizer.step()  ->  optimizer.zero_grad()
```

Code pattern: `train_loss = sum(v.mean() for v in model.fm.optimization_terms(losses).values())`
Check: `train_loss.shape` must be `torch.Size([])` and `train_loss.requires_grad` must be True.

Watch the balance: bb_ca (1836) is ~28x bigger than local_latents (65), so the CA term dominates the gradient.
Print each entry's mean separately to see which part is learning. Large numbers (hundreds-thousands) are normal:
the per-structure loss is multiplied by 1 / ((1-t)^2 + 1e-5), so samples with t near 1 weigh a lot.

### Per-structure FM loss, one data mode (rdn_flow_matcher.py:356)
```
x1_pred = x_t + (1 - t) * v                      network velocity v -> clean guess
err     = (x_1 - x1_pred) * supervision_mask     only real, supervised tokens
loss_i  = sum(err^2) / n_supervised_tokens
loss_i  = loss_i * 1 / ((1 - t)^2 + 1e-5)        time weight (t near 1 counts most)
```
Training loop order: zero_grad -> call_nn -> compute_loss -> reduce -> backward -> step.
Optimizer: create ONCE before the loop (AdamW over p.requires_grad params of model.nn, lr ~1e-4).

## Adding a NEW residue token to the vocabulary (147 -> 148)   [PLAN, nothing verified by running yet]

Vocabulary object: `UNIFIED_ATOM37_ENCODING` (rfproteina.encoding / atomworks.ml.encoding_definitions)
has `.tokens`, `.token_to_idx`, `.n_tokens` (= 147). NCAA list = `NCAA_NAMES`.
FOUND (copied into the project): rfproteina_encoding.py (_load_vocabulary line 13, UNIFIED_ATOM37_ENCODING line 39, NCAA_NAMES line 40)
reads aa_rna_dna_glycan_ncaa.yaml (installed: rfproteina/resources/). yaml: expected_num_classes 147 (line 7), ncaa: section line 53,
one line per residue = name -> 37-slot atom layout ("-" = empty slot); ORDER of lines = class id (yaml line 10 says "checkpoint class order").

```
Everything sized by n_tokens (breaks when the vocab grows)
├── data:         residue_type ids made in pipeline.py (ConvertToAtomLevelProteinFeatures);
│                 unknown name -> atomized (id 32, token_type 3)
├── decoder IN:   cond_residue_type one-hot, 147 of the 214 dims   (rfproteina_feature_factory.py:409)
├── decoder OUT:  logit_linear 768 -> 147                          (asserts at rfproteina_autoencoder.py:89, :529)
├── nn:           sequence_readout, 147 classes (detached)
├── pad value:    "<M>" mask id                                    (rfproteina_schema_proteina.py:62)
└── loss groups:  VALIDATION_RESIDUE_GROUPS / _RESIDUE_GROUP_IDS   (rfproteina_autoencoder.py:30-45)
                  a new token lands in the "other" group unless added to a group
```

```
Options (cheapest first)
1. REUSE a rare NCAA slot   no shape change; slot keeps its old pretrained meaning
2. APPEND token (148)       add name + atom37 layout (<= 37 atoms, atom names in the yaml layout)
                            grow Linear/Embedding by 1 row/col, COPY old weights,
                            init new row from a similar residue (parent AA) or the mean row, NOT random
                            train new row + LoRA (modules_to_save = logit_linear, sequence_readout)
3. KEEP ATOMIZING           >37 atoms / odd layout: no vocab change, but no residue latent z
```

Order to code it
```
1. find the encoding file; add one name to a COPY of NCAA_NAMES; print n_tokens
2. run pipeline on one file: residue_type must be the new id, token_type 6 (not 32 / 3)
3. write resize_linear / resize_embedding that keep old weights; test old outputs unchanged
4. apply to decoder logit_linear + sequence_readout (+ any one-hot sized by n_tokens); fix the 2 asserts
5. add the token to a loss group
```
Open question: is my residue a CCD-coded NCAA with <= 37 atoms? yes -> option 2; no -> option 3.
Still unchecked: how many of the 31,911 files are atomized vs real NCAA tokens.

## WHAT TO MODIFY to teach the model new non-canonical side chains  [plan; tensor names/shapes read from model.safetensors header, rest from code]

Checkpoint: RFD4_bundle/model.safetensors (loaded fine_tuning.py:14). Tensors sized by the vocabulary (147):

```
MODULE (checkpoint key)                                           shape        change for 148 tokens          trainable?
----------------------------------------------------------------  -----------  -----------------------------  -------------------------
A autoencoder.decoder.logit_linear.1.weight                       [147, 768]   add 1 ROW per new token        YES (modules_to_save)
B nn.sequence_readout.1.weight                                    [147, 768]   add 1 ROW per new token        YES (modules_to_save)
C autoencoder.decoder.init_repr_factory.feat_creators.3
                                    .embedding.weight             [147, 147]   -> [148,148]; keep identity    no (frozen identity)
D autoencoder.decoder.init_repr_factory.linear_out.weight         [768, 214]   214 -> 215 (one-hot grows)     YES, add 1 COLUMN, init 0
E token_type tables (3 separate copies, [7,32] each; inferred by shape):
    autoencoder.encoder.init_repr_factory.feat_creators.0         [7, 32]      none if token_type stays 6     optional
    autoencoder.decoder.init_repr_factory.feat_creators.2         [7, 32]
    nn.init_repr_factory.feat_creators.2                          [7, 32]
```
(D follows from C: the one-hot is 147 of the 214 decoder-input dims, so one more class = one more input column.)

NOT in the checkpoint (built from code at startup) -> edit the vocabulary SOURCE, not weights:
```
F UNIFIED_ATOM37_ENCODING / NCAA_NAMES (rfproteina.encoding, file not found yet)    new name + its atom37 atom-name layout
G decoder.residue_atom_mask [147, 37]  (autoencoder.py:159, :576)                   rebuilt from F: which of 37 slots the residue may use
H RESIDUES_BY_TOKEN_TYPE (used pipeline.py:292-294)                                  name must be in the "ncaa" group -> token_type 6, not atomized
I VALIDATION_RESIDUE_GROUPS["ncaa"] (autoencoder.py:30-45)                          loss-weight group (else "other")
J the 2 asserts autoencoder.py:89, :529 and cfg nn_ae.n_residue_types (147 -> 148)
```

Modules that do NOT need a shape change (side chain geometry is already generic):
```
encoder input 1587 (37 slots x coords/element/charge/bonds), decoder struct_linear 768->111 and
atom_presence_linear 768->37, latent z (32-d), nn ca_linear / local_latents_linear.
A new side chain with <= 37 atoms just fills slots in the existing 37-slot layout.
```

Side-chain LEARNING (the part LoRA on nn alone does not give):
```
- the 32-d latent z carries the side-chain detail: ENCODER writes it, DECODER turns (CA, z, token_type, one-hot) -> 37 atoms.
- my fine_tuning.py freezes the autoencoder -> decoder cannot learn to draw a NEW side chain.
  => either LoRA/unfreeze the decoder transformer_layers (+ A, D), or keep atomizing (token_type 3).
- nn (denoiser) LoRA on to_qkv/to_out_node teaches it to generate z for the new residue; needs the AE to decode it.
```

Init for new rows (never random): A, B new row <- row of the parent residue (or mean of NCAA rows); D new column <- 0
(or copy parent column); resize by copying old weights, then check old outputs unchanged.

### EXPERIMENT 1: separate small parameter on ONE head (isolated, no vocabulary change)  [plan; not run yet]

Target: decoder.logit_linear = Sequential(LayerNorm, Linear 768 -> 147, no bias)   (rfproteina_decoder.py:147-149)
Wrap ONLY the Linear (index 1).  New module ExtendedLinear(old_linear, n_new):

```
x  [B,L,768]  (output of the LayerNorm)
 ├── old(x)                       [B,L,147]   old Linear, requires_grad = False   (frozen, untouched)
 ├── x @ new_weight.T             [B,L,n_new] new nn.Parameter [n_new,768], init = a parent residue's row
 └── torch.cat([old, new], -1)    [B,L,147+n_new]
```
Checks that must pass:
1. out[..., :147] equals the original head's output exactly (torch.equal / allclose)
2. out.shape[-1] == 147 + n_new
3. only new_weight has requires_grad=True;  after one backward, old weight .grad is None
4. optimizer built from p.requires_grad params only (and weight_decay applies to new_weight only)

HOW the output dim changes: model.autoencoder.decoder.logit_linear[1] = ExtendedLinear(old, n_new)   (Sequential allows item assignment)
Everything on the OUTPUT side that still assumes 147 after that (read from code, not run):
  rfproteina_decoder.py:219-220  logits_out.masked_fill(~allowed_per_token) : mask is [7,147] (built :27) vs logits [b,n,148] -> shape error
                                 => widen allowed_residue_types by 1 column (True for token_type 6 / unknown row)
  rfproteina_decoder.py:237      residue_atom_mask[aatype_max]  : table [147,37]; argmax = 147 -> IndexError  => add a row (new residue's atom slots)
  rfproteina_decoder.py:52-55    n_residue_types must equal UNIFIED_ATOM37_ENCODING.n_tokens (raises at construction)
  rfproteina_autoencoder.py:529  assert n_residue_types == n_tokens   (cross_entropy itself is fine with 148 logits if targets < 148)
  nn.sequence_readout            separate [147,768] head in the denoiser, same treatment
Input side (not part of Experiment 1): one-hot Embedding(147,147) + linear_out 214 -> 215.

## How a residue's SEQUENCE is encoded -> latent -> decoded  (worked example: AIB, yaml line 72)

Read-from-code map. "verified" = read in the code; (?) = inferred, not run.

```
AIB layout (aa_rna_dna_glycan_ncaa.yaml:72):  "N CA C - O - - ... CB1 - - CB2 ... OXT"   37 entries
occupied slots:  0 N | 1 CA | 2 C | 4 O | 26 CB1 | 29 CB2 | 36 OXT      (slot 3 CB is EMPTY for AIB)
residue id    :  AIB = its position in the yaml vocabulary (ids 0..146; NLE was 55 by file order (?), confirm
                 with UNIFIED_ATOM37_ENCODING.token_to_idx["AIB"])
```

### A. Data side: name -> slots + id   (pipeline.py)
```
CIF atom_array: each atom has res_name ("AIB") and atom_name ("CB1")
 (1) residue id : pipeline.py:283  atom_array_to_encoded_resnames(aa, UNIFIED_ATOM37_ENCODING, atomize_token="<A>")
                  -> residue_type [L] int.  name NOT in yaml -> "<A>" (atomized)
 (2) token type : pipeline.py ~290  res_name in yaml group "ncaa" -> token_type 6   (aa 0 rna 1 dna 2 atomized 3 unknown 4 glycan 5 ncaa 6)
 (3) atom slot  : token_encoding.py:54 build_atom37_slots  key = "AIB|CB2" -> slot 29  (key table built from the yaml,
                  token_encoding.py:15; LOCAL_LATENT_RESIDUES = every yaml token except <A>,<G>, constants.py:7)
                  no key -> -1 -> ValueError "Atoms cannot be represented in Atom37"   (atomized atoms all go to slot 1)
 (4) ConvertToAtom37Format (pipeline.py:477) scatters:  coords[token, slot] = xyz ; atom37_mask[token, slot] = True
```
Result for one AIB token:  coords [37,3] (rows 0,1,2,4,26,29,36 filled, rest 0), atom37_mask [37] True at those 7 slots.
A slot is a POSITION in the 37 axis, not a stored number. residue id and slot are separate tensors (nothing is concatenated).

### B. Encoder: the ID is NOT an input; identity is inferred from the mask pattern
```
encoder input (1587 per token) = token_type emb 32 + coords/mask feats 148+148 + chain 1 + bond counts 370 + element/charge 888
  (residue_type is NOT in this list, per the table above)  -> 12 transformer layers -> mean, log_scale -> z_latent [B,L,32]
```
So the encoder tells AIB from NLE only by WHICH slots are filled (+ element/charge/bonds). The yaml header says slots are
chosen to maximise pairwise mask Hamming distance for exactly this reason. A new residue needs a mask pattern unlike all others.

### C. Decoder: sequence is re-derived from z_latent   (rfproteina_decoder.py)
```
decoder input per token (214 -> Linear -> 768), feats_seq in nn_130m_unified.yaml:
   ca_coors_nm 3 + z_latent_seq 32 + token_type emb 32 + cond_residue_type 147 (one-hot of a CONDITIONED sequence)
   cond_residue_type is masked by C_SEQ -> carries signal only on the conditioned fraction; otherwise the decoder
   must read the residue identity out of z_latent.
-> 12 transformer layers -> seqs [B,L,768], then three heads:
   struct_linear        768 -> 111 (37x3)  coors_a37_nm, CA slot fixed (rfproteina_decoder.py ~226)
   atom_presence_linear 768 -> 37          which slots exist
   logit_linear         LayerNorm + Linear(768,147,no bias)   (:147-149, forward :206)  residue logits [B,L,147]
```
### D. Turning logits into the decoded residue and atoms
```
logits * emit_mask                                          (:206)
masked_fill(~allowed_residue_types[token_type])             (:209-220)  table [7,147]: which residues each token_type may output
aatype_max = argmax(logits, -1) * emit_mask                 (:233)      -> residue id
atom_mask  = residue_atom_mask[aatype_max] & emit_mask      (:237)      table [147,37]: legal slots of that residue (from the yaml layout)
atom_mask  = atom_mask & (atom_presence_logits > 0), slot 1 forced True (:240-242)
AutoEncoder.decode (rfproteina_autoencoder.py:140): if C_SEQ conditioned token (not <M>) -> use given residue id instead of argmax
```
Loss (AE training_step, rfproteina_autoencoder.py:~176): kl + struct_l1/l2 + atom_presence_bce + ce_seq(logits vs batch["residue_type"]).

### E. What a NEW residue (id 147+) must change  (all 147-wide pieces)
```
yaml entry (name + 37-slot layout; <=37 entries, unique atom names, slot1=CA, slot36=OXT) -> n_tokens 148+
config n_residue_types (nn_130m_unified.yaml:9)  must == n_tokens   (asserts: decoder :52-55, autoencoder.py:89 and :529)
logit_linear[1]       [147,768] -> [n,768]         (your ExtendedLinear, test.py)
allowed_residue_types [7,147]  -> [7,n]            (built at import from the yaml, decoder.py:15-31; token type 6 True for new ids)
residue_atom_mask     [147,37] -> [n,37]           (get_atom_mask(), built from the yaml layout)
cond_residue_type one-hot 147 -> n ; init_repr Linear 214 -> 214+(n-147)
nn.sequence_readout   [147,768] -> [n,768]         (denoiser side)
```
allowed_residue_types / residue_atom_mask are rebuilt from the yaml at construction (persistent=False), so they follow the
yaml automatically IF the model is constructed after the yaml is edited; the learned weights above must be resized by hand.

## Decoder vocabulary-dependent tensors: 147 -> 176  (checked on RFD4_bundle/model.safetensors)

Only THREE autoencoder tensors depend on the vocabulary size. The encoder has none (no 147 or 214 dim).
New vocabulary: aa_rna_dna_glycan_ncaa_ext_v2.yaml = 176 tokens (ids 0-146 unchanged, 29 new residues = ids 147-175,
group `ncaa_ext`, token type 6). Loaded via local_vocab.py (injects rfproteina_encoding.py as `rfproteina.encoding`).

```
per token, decoder input (rfproteina_decoder.py, feats_seq in nn_130m_unified.yaml):
  ca_coors_nm       [3]
  z_latent          [32]
  token_type emb    [32]                      7 types, unchanged
  cond_residue_type: id -> Embedding (1)      [147,147] -> [176,176]   (verified: exactly the identity matrix = a one-hot lookup)
          |
          v  concat  3+32+32+147 = 214   ->   3+32+32+176 = 243
  linear_out (2)   [768,214] -> [768,243]     Linear weight is [out,in]; first 67 cols = CA,z,token_type, last cols = one-hot
          v
  768 per token -> 12 transformer layers (unchanged) -> seqs [B,L,768]
          |- struct_linear 768->111            unchanged
          |- atom_presence_linear 768->37      unchanged
          '- logit_linear = LayerNorm + Linear (3)  [147,768] -> [176,768]  -> logits [B,L,176]
```

| tensor (state-dict name) | now | needs | init of the new part |
|---|---|---|---|
| (1) autoencoder.decoder.init_repr_factory.feat_creators.3.embedding.weight | [147,147] | [176,176] | copy old into top-left corner; extend the identity (new id k -> 1 in new column k) |
| (2) autoencoder.decoder.init_repr_factory.linear_out.weight | [768,214] | [768,243] | copy old 214 cols; ZEROS in the 29 new cols (old behaviour unchanged at start) |
| (3) autoencoder.decoder.logit_linear.1.weight | [147,768] | [176,768] | copy old 147 rows; new rows = copy of a similar residue's row |

Notes
- (1) and (2) are the INPUT side and change together (one-hot width sets linear_out's column count). (3) is the OUTPUT side.
- The new one-hot columns are the LAST ones, and the new ids are appended at the end (147-175), so nothing in the old tensors shifts.
- Encoder: no weight change. Its input is atom37 coords/mask/element/charge/bond counts/token_type, never the residue id.
  Only the config value encoder.n_residue_types (147 -> 176) changes (size check at autoencoder.py:89).
  Open risk: the encoder never saw these slot patterns, so check reconstruction before deciding to fine-tune it.
- config.json has n_residue_types = 147 in 3 places: nn_ae, nn_ae.encoder, nn_ae.decoder (autoencoder_config).
  Proteina.from_pretrained builds the model from the config then load_state_dict(strict=True), so resize the state dict
  and the config BEFORE building (rfproteina.utils.hub.load_pretrained_bundle returns payload["config"] and the state).
- The denoiser side (nn.sequence_readout [147,768], etc.) is NOT part of the autoencoder work.
- Training pool: /mnt/c/Research/Dataset_ncaa/training_pool_all (1612 cif, Boltz-2 predictions, + 1 tsv; use glob("*.cif")).
  Key check passed on it AND on all_structures: 0 atoms missing from layouts, 0 residues outside the vocabulary.
  Rare new residues in the pool: 1AC (3 files), STA (19), XCP (21); D-amino acids / N-methyl are common (~360-670 files).
