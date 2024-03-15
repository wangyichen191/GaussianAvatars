from torch import nn
import torch
import numpy as np
import math
import torch.nn.functional as F
from pytorch3d.io import load_obj
from utils.general_utils import Embedder



class Deform_Model(nn.Module):
    def __init__(self):
        super().__init__()

        # positional encoding
        self.pts_freq = 8
        self.pts_embedder = Embedder(self.pts_freq)

        self.deformNet = MLP(
            input_dim=self.pts_embedder.dim_embeded+100,
            output_dim=13,
            hidden_dim=256,
            hidden_layers=6
        )

    def forward(self, verts_cano, flame_params):
        verts_cano_embedded = self.pts_embedder(verts_cano) # (1, 5143, 51)
        flame_params = flame_params.unsqueeze(0).repeat(1, verts_cano_embedded.shape[1], 1)
        conditioned_verts_cano_embedded = torch.cat((verts_cano_embedded, flame_params), dim=2)
        deforms = self.deformNet(conditioned_verts_cano_embedded)
        return deforms

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers-1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        # input: B,V,d
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size*N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs):
            h = self.fcs[i](h)
            h = F.relu(h)
        output = self.output_linear(h)
        output = output.reshape(batch_size, N_v, -1)

        return output

