import os
from typing import List, Union

import torch
import torch.distributed as dist


def is_master() -> bool:
    return get_dist_rank() == 0


def list_sum(x: list) -> any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def list_mean(x: list) -> any:
    return list_sum(x) / len(x)


def get_dist_rank() -> int:
    return int(os.environ["RANK"])


def get_dist_size() -> int:
    return int(os.environ["WORLD_SIZE"])


@torch.no_grad()
def sync_tensor(
    tensor: Union[torch.Tensor, float], reduce="mean"
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.Tensor(1).fill_(tensor).cuda()
    tensor_list = [torch.empty_like(tensor) for _ in range(get_dist_size())]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == "mean":
        return list_mean(tensor_list)
    elif reduce == "sum":
        return list_sum(tensor_list)
    elif reduce == "cat":
        return torch.cat(tensor_list, dim=0)
    elif reduce == "root":
        return tensor_list[0]
    else:
        return tensor_list


@torch.no_grad()
def all_gather_cat(world_size, tensor, dim=0):
    if world_size == 1:
        return tensor

    g_tensor = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(g_tensor, tensor)
    g_tensor = torch.cat(g_tensor, dim=dim)

    return g_tensor
