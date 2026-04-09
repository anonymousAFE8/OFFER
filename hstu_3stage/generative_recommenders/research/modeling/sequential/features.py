import logging
from typing import Dict, NamedTuple, Optional, Tuple

import torch.nn.functional as F

import torch


class SequentialFeatures(NamedTuple):
    past_lengths: torch.Tensor
    past_ids: torch.Tensor
    past_embeddings: Optional[torch.Tensor]
    past_payloads: Dict[str, torch.Tensor]
    user_id: torch.Tensor


def seq_features_from_row(
    row: Dict[str, torch.Tensor],
    device: int,
    max_output_length: int,
) -> Tuple[SequentialFeatures, torch.Tensor, torch.Tensor]:
    historical_lengths = row["seq_len"].to(device)  # [B]
    historical_ids = row["seq_ids"].to(device)  # [B, N]
    historical_ratings = row["actions"].to(device)
    target_ids = row["target_ids"].to(device).unsqueeze(1)  # [B, 1]
    target_ratings = row["target_actions"].to(device).unsqueeze(1)
    if "timestamps" in row.keys():
        historical_timestamps = row["timestamps"].to(device)
        target_timestamps = row["target_timestamps"].to(device).unsqueeze(1)
    else:
        historical_timestamps, target_timestamps = None, None
    if max_output_length > 0:
        B = historical_lengths.size(0)
        historical_ids = torch.cat(
            [
                historical_ids,
                torch.zeros(
                    (B, max_output_length), dtype=historical_ids.dtype, device=device
                ),
            ],
            dim=1,
        )
        historical_ratings = torch.cat(
            [
                historical_ratings,
                torch.zeros(
                    (B, max_output_length),
                    dtype=historical_ratings.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        if historical_timestamps is not None:
            historical_timestamps = torch.cat(
                [
                    historical_timestamps,
                    torch.zeros(
                        (B, max_output_length),
                        dtype=historical_timestamps.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            historical_timestamps.scatter_(
                dim=1,
                index=historical_lengths.view(-1, 1),
                src=target_timestamps.view(-1, 1),
            )
        # print(f"historical_ids.size()={historical_ids.size()}, historical_timestamps.size()={historical_timestamps.size()}")
    features = SequentialFeatures(
        past_lengths=historical_lengths,
        past_ids=historical_ids,
        past_embeddings=None,
        past_payloads={
            "timestamps": historical_timestamps,
            "ratings": historical_ratings,
        },
        # 加入用户id
        user_id=row["user_id"].to(device)
    )
    return features, target_ids, target_ratings


def packed_rank_features_from_row(
    row: Dict[str, torch.Tensor],
    device: int,
    max_output_length: int,
) -> Tuple[
    SequentialFeatures,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    historical_lengths = row["seq_len"].to(device)
    historical_ids = row["seq_ids"].to(device)
    historical_ratings = row["actions"].to(device)
    candidate_ids = row["candidate_ids"].to(device)
    candidate_ratings = row["candidate_actions"].to(device)
    candidate_mask = row["candidate_mask"].to(device)
    target_ids = row["target_ids"].to(device).unsqueeze(1)
    target_ratings = row["target_actions"].to(device).unsqueeze(1)
    target_timestamps = None
    historical_timestamps = None
    if "timestamps" in row.keys():
        historical_timestamps = row["timestamps"].to(device)
        target_timestamps = row["target_timestamps"].to(device).unsqueeze(1)
    if max_output_length > 0:
        batch_size = historical_lengths.size(0)
        historical_ids = torch.cat(
            [
                historical_ids,
                torch.zeros(
                    (batch_size, max_output_length),
                    dtype=historical_ids.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        historical_ratings = torch.cat(
            [
                historical_ratings,
                torch.zeros(
                    (batch_size, max_output_length),
                    dtype=historical_ratings.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        if historical_timestamps is not None:
            historical_timestamps = torch.cat(
                [
                    historical_timestamps,
                    torch.zeros(
                        (batch_size, max_output_length),
                        dtype=historical_timestamps.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
    features = SequentialFeatures(
        past_lengths=historical_lengths,
        past_ids=historical_ids,
        past_embeddings=None,
        past_payloads={
            "timestamps": historical_timestamps,
            "ratings": historical_ratings,
        },
        user_id=row["user_id"].to(device),
    )
    return (
        features,
        target_ids,
        target_ratings,
        candidate_ids,
        candidate_ratings,
        candidate_mask.bool(),
        target_timestamps,
    )


def expand_packed_rank_candidates(
    seq_features: SequentialFeatures,
    candidate_ids: torch.Tensor,
    candidate_ratings: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_timestamps: Optional[torch.Tensor],
) -> Tuple[SequentialFeatures, torch.Tensor, torch.Tensor]:
    batch_size, num_candidates = candidate_ids.shape
    expanded_lengths = seq_features.past_lengths.repeat_interleave(num_candidates)
    expanded_ids = seq_features.past_ids.repeat_interleave(num_candidates, dim=0)
    expanded_ratings = seq_features.past_payloads["ratings"].repeat_interleave(num_candidates, dim=0)
    expanded_payloads: Dict[str, Optional[torch.Tensor]] = {
        "ratings": expanded_ratings,
        "timestamps": None,
    }
    expanded_timestamps = None
    if seq_features.past_payloads["timestamps"] is not None:
        expanded_timestamps = seq_features.past_payloads["timestamps"].repeat_interleave(
            num_candidates,
            dim=0,
        )
        expanded_payloads["timestamps"] = expanded_timestamps
    expanded_features = SequentialFeatures(
        past_lengths=expanded_lengths,
        past_ids=expanded_ids,
        past_embeddings=None,
        past_payloads=expanded_payloads,
        user_id=seq_features.user_id.repeat_interleave(num_candidates),
    )
    flat_candidate_ids = candidate_ids.reshape(batch_size * num_candidates, 1)
    flat_candidate_ratings = candidate_ratings.reshape(batch_size * num_candidates, 1)
    flat_candidate_mask = candidate_mask.reshape(batch_size * num_candidates)
    expanded_features.past_ids.scatter_(
        dim=1,
        index=expanded_features.past_lengths.view(-1, 1),
        src=flat_candidate_ids,
    )
    expanded_features.past_payloads["ratings"].scatter_(
        dim=1,
        index=expanded_features.past_lengths.view(-1, 1),
        src=flat_candidate_ratings,
    )
    if expanded_timestamps is not None and target_timestamps is not None:
        expanded_target_timestamps = target_timestamps.repeat_interleave(num_candidates, dim=0)
        expanded_timestamps.scatter_(
            dim=1,
            index=expanded_features.past_lengths.view(-1, 1),
            src=expanded_target_timestamps,
        )
    expanded_features.past_lengths.add_(1)
    return expanded_features, flat_candidate_ratings, flat_candidate_mask

