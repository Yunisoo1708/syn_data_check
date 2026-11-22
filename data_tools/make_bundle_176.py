from safetensors.torch import load_file, save_file
from pathlib import Path
import torch

weights = load_file(Path(__file__).resolve().parent.parent/'models'/'RFD4_bundle_176'/'model.safetensors')

weights['autoencoder.decoder.init_repr_factory.feat_creators.3.embedding.weight'] = torch.eye(176, dtype=weights['autoencoder.decoder.init_repr_factory.feat_creators.3.embedding.weight'].dtype).requires_grad_(False)

old_linear_out = weights['autoencoder.decoder.init_repr_factory.linear_out.weight']
new_linear_out = torch.zeros(768, 243, dtype=old_linear_out.dtype) #Remember on .nn has weights associated with it.
new_linear_out[:, :214] = old_linear_out
weights['autoencoder.decoder.init_repr_factory.linear_out.weight'] = new_linear_out

old_logit = weights['autoencoder.decoder.logit_linear.1.weight']
new_logit = torch.zeros(176, 768, dtype=old_logit.dtype) #Torch stores weight as ouput, input format. Remmeber.
new_logit[:147] = old_logit #New logit has shape [147,768]
new_logit[147:] = old_logit.mean(dim=0) #initialize all the weight to the average of previous learned weights
weights['autoencoder.decoder.logit_linear.1.weight'] = new_logit

save_file(weights, Path(__file__).resolve().parent.parent/'models'/'RFD4_bundle_176'/'model.safetensors', metadata={"format": "pt"}) #metadata = shows where the source comes from.


