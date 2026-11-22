import rfd4_proteina as rfd
import peft as pt
import biotite.structure.io.pdbx as pdbx 
import torch
from pathlib import Path
from itertools import islice
from functools import partial
from torch.utils.data import ConcatDataset, DataLoader
from pipeline import build_design_train_pipeline
from inference_datasets import DesignProblemDataset
from rfproteina_collate import collate_batch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = rfd.Proteina.from_pretrained('/mnt/c/Research/RFD4_fine_tune/models/RFD4_bundle', device='cuda')
model.train()

#Part 1 (Find all the target weights of denoiser to train)
targets = ['to_qkv', 'to_out_node']
path_to_w_diffusion = [f"transformer_layers.{i}.mhba.mha.{q}" for i in range(14) for q in targets]


#Part 2 set up LoRA config
cfg = pt.LoraConfig(r=16, lora_alpha=32, 
                    target_modules=path_to_w_diffusion, 
                    modules_to_save=['model.nn.ca_linear', 'model.nn.local_latents_linear'], 
                    bias='lora_only')


model.nn = pt.get_peft_model(model.nn, cfg) #returns model with "PEFT object" with places to edit the weights. PEFT searchs modules within "model.nn" as specified so no need in f-string above.

#Part 3 prepare the structure to match the model format.
path_folder = '/mnt/c/Research/Dataset_ncaa/training_pool_all' 
folder = Path(path_folder)

# --- Split the 31,911 files into chunks ---------------------------------------
# DesignProblemDataset slows down quadratically with the number of files it gets
# (one pass over ALL paths per file). Small chunks keep each dataset fast.
N_FILES = 2      # test size; set to None to use every file
CHUNK = 2000     # files per DesignProblemDataset

if N_FILES is None:
    files = sorted(folder.glob('*.cif'))
else:
    files = list(islice(folder.glob('*.cif'), N_FILES))   # stops early, no full listing
chunks = [files[i:i + CHUNK] for i in range(0, len(files), CHUNK)]
print(f"{len(files)} files -> {len(chunks)} chunk(s)")

pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None)
# One dataset per chunk, then glued together so ds[i] works across all chunks.
ds = ConcatDataset([DesignProblemDataset(c, transform=pipe) for c in chunks])

# Schema = Rulebook for padding (Follow proteindatasample currently). Require knowledge: Which dimension is length, Which value fills the padding, Which dtype to use
SCHEMA = 'rfproteina.datapipes.schema.proteina.ProteinaDataSample'

# DataLoader takes `batch_size` items from ds and hands them to collate_batch. Batch_size = 3 structures per time. Collate_batch stacked the feat.
loader = DataLoader(ds, batch_size=3, shuffle=False,
                    collate_fn=partial(collate_batch, schema=SCHEMA))

epoch = 5
optimizer = torch.optim.AdamW(model.nn.parameters(), lr=1e-4) #For AdamW remember to create a before running the trial otherwise you will lose the running average that it has been accumulating.

for training_batch in loader:
    batch_cuda = {k: v.to(device) for k, v in training_batch.items() if isinstance(v, torch.Tensor)}

    batch_cuda['coords_nm'] = batch_cuda['coords']/10
    batch_cuda['coord_mask'] = batch_cuda['atom37_mask']
    
    #Part 4: Encoding the module.
    with torch.no_grad():
        encoder = model.autoencoder.encoder 
        data = encoder(batch_cuda) #Return data as a dict. 
        model.add_clean_samples(batch_cuda) #Returns the C_a coordinates and latent representation of X_1

    model.fm.corrupt_batch(batch_cuda) #Modify on batch_cuda to get noised samples. 
    #Training loop
    for _ in range(epoch):
        optimizer.zero_grad()
        nn_out = model.call_nn(batch_cuda, n_recycle=1) #Can change n_recycle to higher number but cost more. 
        dict_train_loss = model.fm.compute_loss(batch_cuda, nn_out) #Punish error near 1 whereas lower weights in those that is further from 1. Include other peripheral loss as well.
        dict_train_loss = model.fm.optimization_terms(dict_train_loss) #Shortened the loss dict to only bb_ca and local_latents losses.
        train_loss = sum(v for v in dict_train_loss.values()).mean()
        print(f"Train loss: {train_loss}")
        train_loss.backward() #Remember backward requires 1 number as the loss.
        optimizer.step() #Remmeber optimizer are those that translate a number into a matrix of gradients




