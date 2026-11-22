import local_vocab #MUST be the first import. It makes every rfproteina module use the 176-token vocabulary (ext_v2 yaml).
import rfd4_proteina as rfd
import os, time, random
import torch
import torch.nn.functional as F
from pathlib import Path
from functools import partial
from torch.utils.data import ConcatDataset, DataLoader
from pipeline import build_design_train_pipeline
from inference_datasets import DesignProblemDataset
from rfproteina_collate import collate_batch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

##Part 1: Load the resized model (bundle_176 = same weights as RFD4_bundle, but 3 decoder tables are bigger: 147 -> 176).
model = rfd.Proteina.from_pretrained(os.environ.get('RFD_BUNDLE', '/mnt/c/Research/RFD4_fine_tune/models/RFD4_bundle_176'), device=device) #Env var lets the cluster job point at its own copy.
encoder = model.autoencoder.encoder #encoder has the forward method so we can call it like the usual function.
decoder = model.autoencoder.decoder

##Part 2: Freeze everything, then unfreeze ONLY the 2 tables that must learn the new residues.
for p in model.parameters():
    p.requires_grad_(False) #requires_grad False = the optimizer cannot change it.
# The embedding table (one-hot, identity matrix) stays frozen on purpose: nothing to learn there.
decoder.init_repr_factory.linear_out.weight.requires_grad_(True) #[768, 243]: the 29 new columns start at zero.
decoder.logit_linear[1].weight.requires_grad_(True) #[176, 768]: the 29 new rows start as the average of the old rows.
RESUME = os.environ.get('RESUME') #Path of a saved best_decoder.pt: continue from those 2 tables instead of the fresh bundle values.
if RESUME:
    model.load_state_dict(torch.load(RESUME, map_location=device), strict=False) #strict=False: the file holds only the 2 unfrozen tables.
    print(f'resumed from {RESUME}', flush=True)
SAVE = os.environ.get('SAVE_PATH', str(Path(__file__).parent / 'best_decoder.pt')) #Where the best weights go (set a new name so a resume never overwrites the old file).
params = [p for p in model.parameters() if p.requires_grad] #Only what is unfrozen goes to the optimizer.
print(f"trainable tensors: {len(params)}, trainable numbers: {sum(p.numel() for p in params)}") #Expect 2 tensors, 321,792 numbers (768*243 + 176*768).
#weight_decay=0: weight decay shrinks weights even when their gradient is 0, which would slowly change the OLD rows/columns.
optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0) #Create ONCE, before training, otherwise it loses its running averages.

##Part 3: Data. Split the FILES into train / validation / test (80 / 10 / 10).
N_FILES = int(os.environ['N_FILES']) if os.environ.get('N_FILES') else None #test size; set to None to use every file.
BATCH = int(os.environ.get('BATCH', 2000)) #structures per batch. Raise it until you get an out-of-memory error, then halve it.
folder = Path(os.environ.get('DATA_DIR', '/mnt/c/Research/Dataset_ncaa/training_pool_all'))
files = sorted(folder.glob('*.cif')) #'*.cif' skips the .tsv file in the folder. sorted = same order on every run.
random.Random(0).shuffle(files) #Shuffle with a fixed seed: random split, but the SAME split every time you run.
if N_FILES is not None:
    files = files[:N_FILES]
n_val = max(1, int(0.1 * len(files)))
n_test = max(1, int(0.1 * len(files)))
val_files = files[:n_val] #Used after every epoch to decide which model is the best.
test_files = files[n_val:n_val + n_test] #Touched ONCE at the very end, to report the final score.
train_files = files[n_val + n_test:] #The only files the model learns from.
print(f"files: train {len(train_files)}, val {len(val_files)}, test {len(test_files)}")

pipe = build_design_train_pipeline(crop_size=None, crop_size_range=None) #Pipeline that turns one CIF into tensors (residue_type, coords, atom37_mask...).
SCHEMA = 'rfproteina.datapipes.schema.proteina.ProteinaDataSample' #Rulebook for padding.
CHUNK = 250 #files per DesignProblemDataset. Build time grows with the SQUARE of this (measured: 250 files = 0.7 s, 500 = 2.8 s, 2000 = 45 s per chunk), so keep each one small.
def make_loader(file_list, shuffle, workers):
    chunks = [file_list[i:i + CHUNK] for i in range(0, len(file_list), CHUNK)] #Cut the list into pieces of CHUNK files.
    ds = ConcatDataset([DesignProblemDataset(c, transform=pipe) for c in chunks]) #One dataset per piece, glued together so ds[i] still works over all files.
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, collate_fn=partial(collate_batch, schema=SCHEMA), #collate pads + stacks BATCH structures.
                      num_workers=workers, #Helper processes that read CIFs in parallel while the GPU trains. 0 = the main process does everything (slow).
                      persistent_workers=True, #Keep the helpers alive between epochs (no restart cost).
                      pin_memory=(device == 'cuda')) #Faster CPU -> GPU copy.
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 8)) #Your PC has 32 cores. If the GPU is still waiting, raise it; if RAM runs low, lower it.
train_loader = make_loader(train_files, shuffle=True, workers=NUM_WORKERS) #Shuffle only the training data.
val_loader = make_loader(val_files, shuffle=False, workers=2) #Small sets need fewer helpers.
test_loader = make_loader(test_files, shuffle=False, workers=2)

##Part 4: One batch -> encoder (frozen) -> decoder (training) -> loss.
def batch_loss(training_batch):
    batch_cuda = {k: v.to(device) for k, v in training_batch.items() if isinstance(v, torch.Tensor)} #Move tensors to the GPU (lists/dicts are dropped).
    batch_cuda['coords_nm'] = batch_cuda['coords'] / 10 #Angstrom -> nanometer. The encoder asserts this key exists.
    batch_cuda['coord_mask'] = batch_cuda['atom37_mask']
    with torch.no_grad(): #Encoder is frozen, so it needs no gradient.
        batch_cuda['z_latent'] = encoder(batch_cuda)['z_latent'] #[B, L, 32]
    batch_cuda['ca_coors_nm'] = batch_cuda['coords_nm'][..., 1, :] #CA is atom slot 1 -> [B, L, 3]
    output = decoder(batch_cuda) #NOT under no_grad: this is the part that learns.
    emit = output['emit_mask'] #[B, L] True = a real residue token that the decoder predicts (padding/atomized = False).
    if emit.sum() == 0:
        return None #Nothing to score in this batch.
    logits = output['seq_logits'][emit] #[N, 176]  N = number of real tokens in the batch
    target = batch_cuda['residue_type'][emit].long() #[N]  whole-number ids 0..175
    return F.cross_entropy(logits, target) #Classification loss. A random model starts near ln(176) = 5.2.

def run_epoch(loader, training):
    model.train() if training else model.eval()
    encoder.eval() #Encoder is always in eval mode (frozen).
    total, count = 0.0, 0
    for training_batch in loader:
        with torch.set_grad_enabled(training): #Gradients only while training; validation/test do not need them.
            loss = batch_loss(training_batch)
        if loss is None:
            continue
        if training:
            optimizer.zero_grad() #Clear old gradients ONCE PER BATCH.
            loss.backward() #Compute gradients for the 2 unfrozen tables.
            optimizer.step() #Update them.
        total += loss.item() #.item() = plain number, keeps no history.
        count += 1
        if training and count <= 3 and device == 'cuda':
            print(f"batch {count}: loss {loss.item():.4f}, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB (batch size {BATCH})", flush=True) #Shows how much room is left to raise BATCH.
    return total / max(count, 1) #Average loss over the batches.

##Part 5: Train, keep the best model (judged on validation files), then report on the test files.
def train(epoch, optimizer: torch.optim.AdamW, train_data: DataLoader, val_data: DataLoader, save_path='best_decoder.pt'):
    Best_loss = run_epoch(val_data, training=False) if RESUME else float('inf') #Resuming: start from the resumed model's own val loss, so a worse epoch cannot overwrite the good file. Fresh start: infinity.
    if RESUME: print(f'val loss of resumed model: {Best_loss:.4f}', flush=True)
    t0 = time.time()
    for ep in range(epoch):
        train_loss = run_epoch(train_data, training=True)
        val_loss = run_epoch(val_data, training=False) #Average over ALL validation batches, compared ONCE per epoch.
        print(f"epoch {ep}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, elapsed {time.time() - t0:.0f}s", flush=True)
        if val_loss < Best_loss:
            Best_loss = val_loss
            trained = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad} #Only the unfrozen tables (a few MB).
            torch.save(trained, save_path)
            print(f'saved best model (val loss {Best_loss:.4f})')
    return Best_loss

if __name__ == '__main__': #Runs only when you start this file, not when another file imports it.
    train(int(os.environ.get('EPOCHS', 5)), optimizer, train_loader, val_loader, save_path=SAVE)
    model.load_state_dict(torch.load(SAVE, map_location=device), strict=False) #Go back to the best weights (strict=False: file holds only 2 tables).
    print(f"test loss (best model): {run_epoch(test_loader, training=False):.4f}")
