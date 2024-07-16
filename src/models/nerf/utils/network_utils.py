import torch
import torch.nn as nn
from typing import Optional, Callable


def batchify(
        network: nn.Module,
        chunk: Optional[int] = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    if chunk is None:
        return network

    def ret(inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([network(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], dim=0)

    return ret


def run_network(
        coords: torch.Tensor,
        viewdirs: Optional[torch.Tensor],
        network: nn.Module,
        coords_encoder: Callable[[torch.Tensor], torch.Tensor],
        viewdirs_encoder: Callable[[torch.Tensor], torch.Tensor],
        net_chunk: int = 1024 * 64
) -> torch.Tensor:

    coords_flat = coords.view(-1, coords.shape[-1])  # [N, N_c/N_f, 3] -> [N * N_c/N_f, 3]
    coords_encoded = coords_encoder(coords_flat)  # [N * N_c/N_f, 3] -> [N * N_c/N_f, F_c]

    if viewdirs is not None:
        viewdirs_ = viewdirs[:, None].expand_as(coords)  # [N, 3] -> [N, 1, 3] -> [N, N_c/N_f, 3]
        viewdirs_flat = viewdirs_.reshape(-1, viewdirs_.shape[-1])  # [N, N_c/N_f, 3] -> [N * N_c/N_f, 3]
        viewdirs_encoded = viewdirs_encoder(viewdirs_flat)  # [N * N_c/N_f, 3] -> [N * N_c/N_f, F_v]
        coords_encoded = torch.cat([coords_encoded, viewdirs_encoded], dim=-1)  # [N * N_c/N_f, F_c + F_v]

    outputs_flat = batchify(network, net_chunk)(coords_encoded)  # [N * N_c/N_f, F_c + F_v] -> [N * N_c/N_f, F]
    return outputs_flat.view(*coords.shape[:-1], outputs_flat.shape[-1])  # [N * N_c/N_f, F] -> [N, N_c/N_f, F]
