import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _pad_view(
    ids: torch.Tensor,
    payloads: Dict[str, Optional[torch.Tensor]],
    max_len: int,
    row_dtype: torch.dtype,
    row_device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    padded_ids = torch.zeros(max_len, dtype=row_dtype, device=row_device)
    padded_ids[: ids.numel()] = ids

    padded_payloads: Dict[str, Optional[torch.Tensor]] = {}
    for key, tensor in payloads.items():
        if tensor is None:
            padded_payloads[key] = None
            continue
        padded_tensor = torch.zeros(max_len, dtype=tensor.dtype, device=tensor.device)
        padded_tensor[: tensor.numel()] = tensor
        padded_payloads[key] = padded_tensor
    return padded_ids, padded_payloads


def _apply_crop(
    ids: torch.Tensor,
    payloads: Dict[str, Optional[torch.Tensor]],
    crop_ratio: float,
    rng: random.Random,
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    length = ids.numel()
    if length <= 1:
        return ids, payloads
    crop_len = max(1, int(round(length * crop_ratio)))
    crop_len = min(length, crop_len)
    start = 0 if crop_len == length else rng.randint(0, length - crop_len)
    crop_indices = torch.arange(start, start + crop_len, device=ids.device)

    new_ids = ids[crop_indices]
    new_payloads = {}
    for key, tensor in payloads.items():
        new_payloads[key] = None if tensor is None else tensor[crop_indices]
    return new_ids, new_payloads


def _apply_mask(
    ids: torch.Tensor,
    payloads: Dict[str, Optional[torch.Tensor]],
    mask_ratio: float,
    rng: random.Random,
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    length = ids.numel()
    if length == 0:
        return ids, payloads
    num_mask = max(1, int(round(length * mask_ratio)))
    num_mask = min(length, num_mask)
    mask_positions = rng.sample(range(length), num_mask)

    new_ids = ids.clone()
    new_payloads = {
        key: None if tensor is None else tensor.clone()
        for key, tensor in payloads.items()
    }
    for pos in mask_positions:
        new_ids[pos] = 0
        for key, tensor in new_payloads.items():
            if tensor is not None:
                tensor[pos] = 0
    return new_ids, new_payloads


def _apply_reorder(
    ids: torch.Tensor,
    payloads: Dict[str, Optional[torch.Tensor]],
    reorder_ratio: float,
    rng: random.Random,
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    length = ids.numel()
    if length <= 1:
        return ids, payloads
    reorder_len = max(2, int(round(length * reorder_ratio)))
    reorder_len = min(length, reorder_len)
    start = 0 if reorder_len == length else rng.randint(0, length - reorder_len)
    block_indices = list(range(start, start + reorder_len))
    shuffled = block_indices[:]
    rng.shuffle(shuffled)

    reorder_index = torch.arange(length, device=ids.device)
    reorder_index[start : start + reorder_len] = torch.tensor(
        shuffled,
        dtype=reorder_index.dtype,
        device=reorder_index.device,
    )

    new_ids = ids[reorder_index]
    new_payloads = {}
    for key, tensor in payloads.items():
        new_payloads[key] = None if tensor is None else tensor[reorder_index]
    return new_ids, new_payloads


def _sample_view(
    ids: torch.Tensor,
    payloads: Dict[str, Optional[torch.Tensor]],
    crop_ratio: float,
    mask_ratio: float,
    reorder_ratio: float,
    rng: random.Random,
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    ops = ("crop", "mask", "reorder")
    op = ops[rng.randrange(len(ops))]
    if op == "crop":
        return _apply_crop(ids, payloads, crop_ratio=crop_ratio, rng=rng)
    if op == "mask":
        return _apply_mask(ids, payloads, mask_ratio=mask_ratio, rng=rng)
    return _apply_reorder(ids, payloads, reorder_ratio=reorder_ratio, rng=rng)


def build_cl4srec_views(
    past_ids: torch.Tensor,
    past_lengths: torch.Tensor,
    past_payloads: Dict[str, Optional[torch.Tensor]],
    crop_ratio: float,
    mask_ratio: float,
    reorder_ratio: float,
    random_seed: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    rng = random.Random(random_seed)
    batch_size, max_len = past_ids.shape

    view_ids = [[], []]
    view_lengths = [[], []]
    view_payloads = [[], []]

    for row_idx in range(batch_size):
        seq_len = int(past_lengths[row_idx].item())
        seq_len = max(1, min(seq_len, max_len))

        valid_ids = past_ids[row_idx, :seq_len]
        valid_payloads = {
            key: None if tensor is None else tensor[row_idx, :seq_len]
            for key, tensor in past_payloads.items()
        }

        for view_idx in range(2):
            aug_ids, aug_payloads = _sample_view(
                ids=valid_ids,
                payloads=valid_payloads,
                crop_ratio=crop_ratio,
                mask_ratio=mask_ratio,
                reorder_ratio=reorder_ratio,
                rng=rng,
            )
            padded_ids, padded_payloads = _pad_view(
                ids=aug_ids,
                payloads=aug_payloads,
                max_len=max_len,
                row_dtype=past_ids.dtype,
                row_device=past_ids.device,
            )
            view_ids[view_idx].append(padded_ids)
            view_lengths[view_idx].append(aug_ids.numel())
            view_payloads[view_idx].append(padded_payloads)

    def _stack_view(view_idx: int) -> Dict[str, torch.Tensor]:
        stacked_payloads: Dict[str, Optional[torch.Tensor]] = {}
        payload_keys = past_payloads.keys()
        for key in payload_keys:
            if past_payloads[key] is None:
                stacked_payloads[key] = None
            else:
                stacked_payloads[key] = torch.stack(
                    [payload[key] for payload in view_payloads[view_idx]],
                    dim=0,
                )
        return {
            "past_ids": torch.stack(view_ids[view_idx], dim=0),
            "past_lengths": torch.tensor(
                view_lengths[view_idx],
                dtype=past_lengths.dtype,
                device=past_lengths.device,
            ),
            "past_payloads": stacked_payloads,
        }

    return _stack_view(0), _stack_view(1)


def cl4srec_infonce_loss(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    anchor_norm = F.normalize(anchor_embeddings, dim=-1)
    positive_norm = F.normalize(positive_embeddings, dim=-1)

    logits = torch.matmul(anchor_norm, positive_norm.transpose(0, 1)) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_a + loss_b)
