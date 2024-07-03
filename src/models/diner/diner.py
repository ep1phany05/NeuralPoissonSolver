import torch
import numpy as np
from torch import nn

from src.models.diner import SineLayer


class DINER(nn.Module):
    def __init__(self, hash_mod=True, hash_table_length=512 * 512, in_features=3, hidden_features=64,
                 hidden_layers=2, out_features=3, outermost_linear=True, first_omega_0=6.0, hidden_omega_0=6.0):
        super(DINER, self).__init__()
        self.hash_mod = hash_mod
        self.table = nn.Parameter(1e-4 * (torch.rand((hash_table_length, in_features)) * 2 - 1), requires_grad=True)

        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        layers += [SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0) for _ in range(hidden_layers)]

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6 / hidden_features) / hidden_omega_0
                final_linear.weight.uniform_(-bound, bound)
            layers.append(final_linear)
        else:
            layers.append(SineLayer(hidden_features, out_features, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        if self.hash_mod:
            output = self.net(self.table)
        else:
            output = self.net(coords)
        output = torch.clamp(output, min=-1.0, max=1.0)
        return {"model_out": output, "table": self.table}

    def load_pretrained(self, model_path, device=None):
        if device:
            self.to(device)
            checkpoint = torch.load(model_path, map_location=device)
        else:
            checkpoint = torch.load(model_path, map_location="cpu")
        self.load_state_dict(checkpoint["net"])
