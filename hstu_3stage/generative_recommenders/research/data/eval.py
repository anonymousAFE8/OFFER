# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import logging
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Union

import torch
import torch.distributed as dist

from generative_recommenders.research.indexing.candidate_index import (
    CandidateIndex,
    TopKModule,
)
from generative_recommenders.research.modeling.sequential.features import (
    SequentialFeatures,
)
from generative_recommenders.research.rails.similarities.module import SimilarityModule
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


@dataclass
class EvalState:
    all_item_ids: Set[int]
    candidate_index: CandidateIndex
    top_k_module: TopKModule
    normalize_embeddings_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


def get_eval_state(
    model: SimilarityModule,
    all_item_ids: List[int],  # [X]
    negatives_sampler: torch.nn.Module,
    top_k_module_fn: Callable[[torch.Tensor, torch.Tensor], TopKModule],
    device,
    exp_conf_dict,
    get_feat,
    feat_emb,
    float_dtype: Optional[torch.dtype] = None,
) -> EvalState:
    if exp_conf_dict.get("use_feat", False):
        eval_negatives_ids = torch.as_tensor(all_item_ids).to(device).unsqueeze(0)
        eval_negatives_feats, feat_names = get_feat(eval_negatives_ids.reshape(-1).cpu(), "item", re_feat_name= True)
        for name in feat_names:
           eval_negatives_feats[name] = torch.tensor(eval_negatives_feats[name], dtype=torch.int64).to(device).unsqueeze(0)
        eval_negative_embeddings = feat_emb(item_ids = eval_negatives_ids, item_actions=None, user_id=None, feat = eval_negatives_feats)
    else:
        eval_negatives_ids = torch.as_tensor(all_item_ids).to(device).unsqueeze(0)  # [1, X]
        eval_negative_embeddings = negatives_sampler.normalize_embeddings(
            model.get_item_embeddings(eval_negatives_ids)
        )
    if float_dtype is not None:
        eval_negative_embeddings = eval_negative_embeddings.to(float_dtype)
    candidates = CandidateIndex(
        ids=eval_negatives_ids,
        embeddings=eval_negative_embeddings,
    )
    return EvalState(
        all_item_ids=set(all_item_ids),
        candidate_index=candidates,
        top_k_module=top_k_module_fn(eval_negative_embeddings, eval_negatives_ids),
        normalize_embeddings_fn=negatives_sampler.normalize_embeddings,
    )


def get_restricted_top_k_outputs(
    query_embeddings: torch.Tensor,
    candidate_ids: torch.Tensor,
    item_embedding_getter: Callable[[torch.Tensor], torch.Tensor],
    k: int,
    normalize_embeddings_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_embeddings = item_embedding_getter(candidate_ids)
    if normalize_embeddings_fn is not None:
        candidate_embeddings = normalize_embeddings_fn(candidate_embeddings)
    scores = torch.sum(query_embeddings.unsqueeze(1) * candidate_embeddings, dim=-1)
    top_k = min(int(k), int(candidate_ids.size(1)))
    top_k_scores, top_k_indices = torch.topk(scores, k=top_k, dim=1)
    top_k_ids = torch.gather(candidate_ids, dim=1, index=top_k_indices)
    return top_k_ids, top_k_scores


def build_eval_output_from_topk_ids(
    eval_top_k_ids: torch.Tensor,
    eval_top_k_prs: torch.Tensor,
    target_ids: torch.Tensor,
    device,
    target_ratings: Optional[torch.Tensor] = None,
    min_positive_rating: int = 4,
    return_topk: bool = False,
    max_k: int = 2500,
) -> Dict[str, Union[float, torch.Tensor]]:
    k = eval_top_k_ids.size(1)
    _, eval_rank_indices = torch.max(
        torch.cat(
            [eval_top_k_ids, target_ids],
            dim=1,
        )
        == target_ids,
        dim=1,
    )
    eval_ranks = torch.where(eval_rank_indices == k, max_k + 1, eval_rank_indices + 1)

    output = {
        "ndcg@1": torch.where(
            eval_ranks <= 1,
            torch.div(1.0, torch.log2(eval_ranks + 1)),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "ndcg@10": torch.where(
            eval_ranks <= 10,
            torch.div(1.0, torch.log2(eval_ranks + 1)),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "ndcg@50": torch.where(
            eval_ranks <= 50,
            torch.div(1.0, torch.log2(eval_ranks + 1)),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "ndcg@100": torch.where(
            eval_ranks <= 100,
            torch.div(1.0, torch.log2(eval_ranks + 1)),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "ndcg@200": torch.where(
            eval_ranks <= 200,
            torch.div(1.0, torch.log2(eval_ranks + 1)),
            torch.zeros(1, dtype=torch.float32, device=device),
        ),
        "hr@1": (eval_ranks <= 1),
        "hr@10": (eval_ranks <= 10),
        "hr@50": (eval_ranks <= 50),
        "hr@100": (eval_ranks <= 100),
        "hr@200": (eval_ranks <= 200),
        "hr@500": (eval_ranks <= 500),
        "hr@1000": (eval_ranks <= 1000),
        "mrr": torch.div(1.0, eval_ranks),
    }
    if target_ratings is not None:
        target_ratings = target_ratings.squeeze(1)
        output["ndcg@10_>=4"] = torch.where(
            eval_ranks[target_ratings >= 4] <= 10,
            torch.div(1.0, torch.log2(eval_ranks[target_ratings >= 4] + 1)),
            torch.zeros(1, dtype=torch.float32, device=device),
        )
        output[f"hr@10_>={min_positive_rating}"] = (
            eval_ranks[target_ratings >= min_positive_rating] <= 10
        )
        output[f"hr@50_>={min_positive_rating}"] = (
            eval_ranks[target_ratings >= min_positive_rating] <= 50
        )
        output[f"mrr_>={min_positive_rating}"] = torch.div(
            1.0, eval_ranks[target_ratings >= min_positive_rating]
        )
    if return_topk:
        output["top_k_ids"] = eval_top_k_ids
        output["top_k_scores"] = eval_top_k_prs
    return output


@torch.inference_mode  # pyre-ignore [56]
def eval_metrics_v2_from_tensors(
    eval_state: EvalState,
    model: SimilarityModule,
    row,
    seq_features: SequentialFeatures,
    target_ids: torch.Tensor,  # [B, 1]
    exp_conf_dict,
    min_positive_rating: int = 4,
    target_ratings: Optional[torch.Tensor] = None,  # [B, 1]
    epoch: Optional[str] = None,
    filter_invalid_ids: bool = True,
    user_max_batch_size: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    return_topk: bool = False,
    candidate_topk_ids: Optional[torch.Tensor] = None,
) -> Dict[str, Union[float, torch.Tensor]]:

    B, _ = target_ids.shape
    device = target_ids.device

    for target_id in target_ids:
        target_id = int(target_id)
        if target_id not in eval_state.all_item_ids:
            print(f"missing target_id {target_id}")
    
    item_features = None
    if exp_conf_dict.get("use_feat"):
        item_features = row["item_feat"]
        for k, v in item_features.items():
            item_features[k] = v.to(device)

    shared_input_embeddings = model.encode(
        past_lengths=seq_features.past_lengths,
        past_ids=seq_features.past_ids,
        past_embeddings=model._embedding_module.get_item_embeddings(item_ids = seq_features.past_ids, item_actions = seq_features.past_payloads["ratings"],user_id = seq_features.user_id, feat = item_features),
        past_payloads=seq_features.past_payloads,
    )
    if dtype is not None:
        shared_input_embeddings = shared_input_embeddings.to(dtype)

    MAX_K = 2500
    if candidate_topk_ids is not None:
        candidate_topk_ids = candidate_topk_ids.to(device)
        k = min(MAX_K, candidate_topk_ids.size(1))
        eval_top_k_ids, eval_top_k_prs = get_restricted_top_k_outputs(
            query_embeddings=shared_input_embeddings,
            candidate_ids=candidate_topk_ids,
            item_embedding_getter=model.get_item_embeddings,
            k=k,
            normalize_embeddings_fn=eval_state.normalize_embeddings_fn,
        )
    else:
        k = min(MAX_K, eval_state.candidate_index.ids.size(1))
        user_max_batch_size = user_max_batch_size or shared_input_embeddings.size(0)
        num_batches = (
            shared_input_embeddings.size(0) + user_max_batch_size - 1
        ) // user_max_batch_size
        eval_top_k_ids_all = []
        eval_top_k_prs_all = []
        for mb in range(num_batches):
            eval_top_k_ids, eval_top_k_prs, _ = (
                eval_state.candidate_index.get_top_k_outputs(
                    query_embeddings=shared_input_embeddings[
                        mb * user_max_batch_size : (mb + 1) * user_max_batch_size, ...
                    ],
                    top_k_module=eval_state.top_k_module,
                    k=k,
                    invalid_ids=(
                        seq_features.past_ids[
                            mb * user_max_batch_size : (mb + 1) * user_max_batch_size, :
                        ]
                        if filter_invalid_ids
                        else None
                    ),
                    return_embeddings=False,
                )
            )
            eval_top_k_ids_all.append(eval_top_k_ids)
            eval_top_k_prs_all.append(eval_top_k_prs)

        if num_batches == 1:
            eval_top_k_ids = eval_top_k_ids_all[0]
            eval_top_k_prs = eval_top_k_prs_all[0]
        else:
            eval_top_k_ids = torch.cat(eval_top_k_ids_all, dim=0)
            eval_top_k_prs = torch.cat(eval_top_k_prs_all, dim=0)

    assert eval_top_k_ids.size(1) == k
    return build_eval_output_from_topk_ids(
        eval_top_k_ids=eval_top_k_ids,
        eval_top_k_prs=eval_top_k_prs,
        target_ids=target_ids,
        device=device,
        target_ratings=target_ratings,
        min_positive_rating=min_positive_rating,
        return_topk=return_topk,
        max_k=MAX_K,
    )


def eval_recall_metrics_from_tensors(
    eval_state: EvalState,
    model: SimilarityModule,
    seq_features: SequentialFeatures,
    user_max_batch_size: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    target_ids = seq_features.past_ids[:, -1].unsqueeze(1)
    filtered_past_ids = seq_features.past_ids.detach().clone()
    filtered_past_ids[:, -1] = torch.zeros_like(target_ids.squeeze(1))
    return eval_metrics_v2_from_tensors(
        eval_state=eval_state,
        model=model,
        seq_features=SequentialFeatures(
            past_lengths=seq_features.past_lengths - 1,
            past_ids=filtered_past_ids,
            past_embeddings=seq_features.past_embeddings,
            past_payloads=seq_features.past_payloads,
        ),
        target_ids=target_ids,
        user_max_batch_size=user_max_batch_size,
        dtype=dtype,
    )


def _avg(x: torch.Tensor, world_size: int) -> torch.Tensor:
    _sum_and_numel = torch.tensor(
        [x.sum(), x.numel()], dtype=torch.float32, device=x.device
    )
    if world_size > 1:
        dist.all_reduce(_sum_and_numel, op=dist.ReduceOp.SUM)
    return _sum_and_numel[0] / _sum_and_numel[1]


def add_to_summary_writer(
    writer: Optional[SummaryWriter],
    batch_id: int,
    metrics: Dict[str, torch.Tensor],
    prefix: str,
    world_size: int,
) -> None:
    for key, values in metrics.items():
        avg_value = _avg(values, world_size)
        if writer is not None:
            writer.add_scalar(f"{prefix}/{key}", avg_value, batch_id)


# @torch.inference_mode  # pyre-ignore [56]
# def eval_metrics_from_tensors_rank(
#     eval_state: EvalState,
#     model: SimilarityModule,
#     seq_features: SequentialFeatures,
#     target_ids: torch.Tensor,  # [B, 1]
#     min_positive_rating: int = 4,
#     target_ratings: Optional[torch.Tensor] = None,  # [B, 1]
#     epoch: Optional[str] = None,
#     filter_invalid_ids: bool = True,
#     user_max_batch_size: Optional[int] = None,
#     dtype: Optional[torch.dtype] = None,
# ) -> Dict[str, Union[float, torch.Tensor]]:
#     """
#     Args:
#         eval_negatives_ids: Optional[Tensor]. If not present, defaults to eval over
#             the entire corpus (`num_items`) excluding all the items that users have
#             seen in the past (historical_ids, target_ids). This is consistent with
#             papers like SASRec and TDM but may not be fair in practice as retrieval
#             modules don't have access to read state during the initial fetch stage.
#         filter_invalid_ids: bool. If true, filters seen ids by default.
#     Returns:
#         keyed metric -> list of values for each example.
#     """
#     B, _ = target_ids.shape     # 128
#     device = target_ids.device  

#     for target_id in target_ids:
#         target_id = int(target_id)
#         if target_id not in eval_state.all_item_ids:
#             print(f"missing target_id {target_id}")

#     # computes ro- part exactly once.
#     # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
#     shared_input_embeddings = model.encode(
#         past_lengths=seq_features.past_lengths,
#         past_ids=seq_features.past_ids,
#         # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
#         past_embeddings=model.get_item_embeddings(seq_features.past_ids),
#         past_payloads=seq_features.past_payloads,
#     )
#     if dtype is not None:
#         shared_input_embeddings = shared_input_embeddings.to(dtype)

#     MAX_K = 2500
#     k = min(MAX_K, eval_state.candidate_index.ids.size(1))
#     user_max_batch_size = user_max_batch_size or shared_input_embeddings.size(0)
#     num_batches = (
#         shared_input_embeddings.size(0) + user_max_batch_size - 1
#     ) // user_max_batch_size
#     eval_top_k_ids_all = []
#     eval_top_k_prs_all = []
#     for mb in range(num_batches):
#         eval_top_k_ids, eval_top_k_prs, _ = (
#             eval_state.candidate_index.get_top_k_outputs(
#                 query_embeddings=shared_input_embeddings[
#                     mb * user_max_batch_size : (mb + 1) * user_max_batch_size, ...
#                 ],
#                 top_k_module=eval_state.top_k_module,
#                 k=k,
#                 invalid_ids=(
#                     seq_features.past_ids[
#                         mb * user_max_batch_size : (mb + 1) * user_max_batch_size, :
#                     ]
#                     if filter_invalid_ids
#                     else None
#                 ),
#                 return_embeddings=False,
#             )
#         )
#         eval_top_k_ids_all.append(eval_top_k_ids)
#         eval_top_k_prs_all.append(eval_top_k_prs)

#     if num_batches == 1:
#         eval_top_k_ids = eval_top_k_ids_all[0]
#         eval_top_k_prs = eval_top_k_prs_all[0]
#     else:
#         eval_top_k_ids = torch.cat(eval_top_k_ids_all, dim=0)
#         eval_top_k_prs = torch.cat(eval_top_k_prs_all, dim=0)

#     assert eval_top_k_ids.size(1) == k
#     _, eval_rank_indices = torch.max(
#         torch.cat(
#             [eval_top_k_ids, target_ids],
#             dim=1,
#         )
#         == target_ids,
#         dim=1,
#     )
#     eval_ranks = torch.where(eval_rank_indices == k, MAX_K + 1, eval_rank_indices + 1)

#     output = {
#         "ndcg@1": torch.where(
#             eval_ranks <= 1,
#             torch.div(1.0, torch.log2(eval_ranks + 1)),
#             torch.zeros(1, dtype=torch.float32, device=device),
#         ),
#         "ndcg@10": torch.where(
#             eval_ranks <= 10,
#             torch.div(1.0, torch.log2(eval_ranks + 1)),
#             torch.zeros(1, dtype=torch.float32, device=device),
#         ),
#         "ndcg@50": torch.where(
#             eval_ranks <= 50,
#             torch.div(1.0, torch.log2(eval_ranks + 1)),
#             torch.zeros(1, dtype=torch.float32, device=device),
#         ),
#         "ndcg@100": torch.where(
#             eval_ranks <= 100,
#             torch.div(1.0, torch.log2(eval_ranks + 1)),
#             torch.zeros(1, dtype=torch.float32, device=device),
#         ),
#         "ndcg@200": torch.where(
#             eval_ranks <= 200,
#             torch.div(1.0, torch.log2(eval_ranks + 1)),
#             torch.zeros(1, dtype=torch.float32, device=device),
#         ),
#         "hr@1": (eval_ranks <= 1),
#         "hr@10": (eval_ranks <= 10),
#         "hr@50": (eval_ranks <= 50),
#         "hr@100": (eval_ranks <= 100),
#         "hr@200": (eval_ranks <= 200),
#         "hr@500": (eval_ranks <= 500),
#         "hr@1000": (eval_ranks <= 1000),
#         "mrr": torch.div(1.0, eval_ranks),
#     }
#     if target_ratings is not None:
#         target_ratings = target_ratings.squeeze(1)  # [B]
#         output["ndcg@10_>=4"] = torch.where(
#             eval_ranks[target_ratings >= 4] <= 10,
#             torch.div(1.0, torch.log2(eval_ranks[target_ratings >= 4] + 1)),
#             torch.zeros(1, dtype=torch.float32, device=device),
#         )
#         output[f"hr@10_>={min_positive_rating}"] = (
#             eval_ranks[target_ratings >= min_positive_rating] <= 10
#         )
#         output[f"hr@50_>={min_positive_rating}"] = (
#             eval_ranks[target_ratings >= min_positive_rating] <= 50
#         )
#         output[f"mrr_>={min_positive_rating}"] = torch.div(
#             1.0, eval_ranks[target_ratings >= min_positive_rating]
#         )

#     return output  # pyre-ignore [7]
