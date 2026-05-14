from typing import Tuple
import itertools
import torch

# NOTE: future versions could consider more supercells.
SUPERCELLS = torch.FloatTensor(list(itertools.product((-1, 0, 1), repeat=3)))


def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, *src.shape[1:])
    index = index.reshape(-1, *([1] * (len(src.shape) - 1))).expand_as(src)
    return out.scatter_add_(0, index, src)


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths


def frac2cart(fcoord, cell):
    return fcoord @ cell


def cart2frac(coord, cell):
    invcell = torch.linalg.inv(cell)
    return coord @ invcell
