import rfd4_proteina as rfd
from pathlib import Path 
import torch 

model = rfd.Proteina.from_pretrained('/mnt/c/Research/RFD4_fine_tune/models/RFD4_bundle', device='cuda')
model.train()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

decoder_logit_linear = model.autoencoder.decoder.logit_linear[1] #First layer is [0] - norm layer - and the second layer is Linear.

##Part 1: Modifying the class of the output of the logit linear for sequence developement
class extended_linear(torch.nn.Module): #Module gives an automatic __call__ and which allows for "to('cuda')" method etc.. A basic Pytorch module function.
    def __init__(self, decoder_logit_linear, n_new, device):
        super().__init__()
        self.old_logit_linear = decoder_logit_linear.requires_grad(False).to(device) #Note that decoder_logit_linear is a Linear object so it needs .weight to show the tensor shape
        self.added_logit_linear = torch.nn.Linear(decoder_logit_linear.in_features, n_new, bias=False).to(device)

    def forward(self, x):
        output_new_ncaa = self.added_logit_linear(x) 
        with torch.no_grad():
            old_output = self.old_logit_linear(x)
        new_output = torch.concatenate([old_output, output_new_ncaa], dim=-1)
        return old_output, new_output 

