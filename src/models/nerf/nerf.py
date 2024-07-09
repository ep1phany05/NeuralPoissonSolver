import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(
            self,
            hidden_layers: int = 8,  # Number of layers for density (sigma) encoder.
            hidden_features: int = 256,  # Number of hidden units in each layer.
            in_features_xyz: int = 63,  # Number of input features for xyz.
            in_features_dir: int = 27,  # Number of input features for direction.
            skips: list = [4],  # Layer index to add skip connection in the Dth layer.
            out_features: int = 4,  # Number of output features.
            use_viewdirs: bool = True  # Whether to use viewing directions.
    ):
        super(NeRF, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.in_features_xyz = in_features_xyz
        self.in_features_dir = in_features_dir
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.xyz_layer = nn.ModuleList(
            [nn.Linear(self.in_features_xyz, hidden_features)] +
            [nn.Linear(hidden_features, hidden_features) if i not in self.skips
             else nn.Linear(hidden_features + self.in_features_xyz, hidden_features) for i in range(hidden_layers - 1)]
        )

        self.dir_layer = nn.Sequential(
            nn.Linear(hidden_features + self.in_features_dir, hidden_features // 2),
            nn.ReLU(True)
        )

        if self.use_viewdirs:
            self.feature_layer = nn.Linear(hidden_features, hidden_features)
            self.alpha_layer = nn.Linear(hidden_features, 1)
            self.rgb_layer = nn.Sequential(nn.Linear(hidden_features // 2, 3), nn.Sigmoid())

        else:
            self.output_layer = nn.Linear(hidden_features, out_features)

    def forward(self, x):

        input_xyz, input_dir = (torch.split(x, [self.in_features_xyz, self.in_features_dir], dim=-1))
        xyz_ = input_xyz
        for i, l in enumerate(self.xyz_layer):
            xyz_ = self.xyz_layer[i](xyz_)
            xyz_ = F.relu(xyz_)
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)

        if self.use_viewdirs:
            alpha = self.alpha_layer(xyz_)
            feature = self.feature_layer(xyz_)
            xyz_ = torch.cat([feature, input_dir], -1)
            xyz_ = self.dir_layer(xyz_)
            rgb = self.rgb_layer(xyz_)
            outputs = torch.cat([rgb, alpha], -1)  # (3+1)=4
        else:
            outputs = self.output_layer(xyz_)

        return outputs

    def load_pretrained(self, model_path, model_type, device=None):
        if model_type not in ["coarse", "fine"]:
            raise ValueError(f"Invalid model type: {model_type}")

        map_location = device if device else "cpu"
        ckpt = torch.load(model_path, map_location=map_location)

        state_dict_key = "network_fn_state_dict" if model_type == "coarse" else "network_fine_state_dict"
        state_dict = {k.replace('module.', ''): v for k, v in ckpt[state_dict_key].items()}

        if device:
            self.to(device)

        self.load_state_dict(state_dict)

