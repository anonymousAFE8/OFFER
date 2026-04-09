from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from generative_recommenders.research.modeling.sequential.autoregressive_losses import (
    AutoregressiveLoss,
    NegativesSampler,
)

from torch.utils.checkpoint import checkpoint


class SampledSoftmaxLoss(AutoregressiveLoss):
    def __init__(
        self,
        num_to_sample: int,
        softmax_temperature: float,
        model,
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self._num_to_sample: int = num_to_sample
        self._softmax_temperature: float = softmax_temperature
        self._model = model
        self._activation_checkpoint: bool = activation_checkpoint       

    def forward(  # pyre-ignore [15]
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,
        supervision_ids: torch.Tensor,
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,
        negatives_sampler: NegativesSampler,
        **kwargs,
    ):
        
        B, N, D = output_embeddings.shape
        output_embeddings = output_embeddings.reshape(-1, D)
        supervision_ids = supervision_ids.reshape(-1)
        supervision_embeddings = supervision_embeddings.reshape(-1, D)
        supervision_weights = supervision_weights.reshape(-1)
        
        sampled_ids, sampled_negative_embeddings = negatives_sampler(
            positive_ids=supervision_ids,
            num_to_sample=self._num_to_sample,
        )
        positive_embeddings = negatives_sampler.normalize_embeddings(
            supervision_embeddings
        )
        positive_logits, aux_losses = self._model.similarity_fn(
            query_embeddings=output_embeddings,  # [B, D] = [N', D]
            item_ids=supervision_ids.unsqueeze(1),  # [N', 1]
            item_embeddings=positive_embeddings.unsqueeze(1),  # [N', D] -> [N', 1, D]
            **kwargs,
        )
        positive_logits = positive_logits / self._softmax_temperature  # [0]
        sampled_negatives_logits, _ = self._model.similarity_fn(
            query_embeddings=output_embeddings,  # [N', D]
            item_ids=sampled_ids,  # [N', R]
            item_embeddings=sampled_negative_embeddings,  # [N', R, D]
            **kwargs,
        )  # [N', R]  # [0]
        sampled_negatives_logits = torch.where(
            supervision_ids.unsqueeze(1) == sampled_ids,  # [N', R]
            -5e4,
            sampled_negatives_logits / self._softmax_temperature,
        )
        sample_softmax_loss = -F.log_softmax(torch.cat([positive_logits, sampled_negatives_logits], dim=1), dim=1)[:, 0]
        return (sample_softmax_loss * supervision_weights).sum() / supervision_weights.sum(), aux_losses



class WeightedFutureSampledSoftmaxLoss(AutoregressiveLoss):
    def __init__(
        self,
        num_to_sample: int,
        softmax_temperature: float,
        model,
        future_window_size: int,       
        decay_rate: float,            
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self._num_to_sample: int = num_to_sample
        self._softmax_temperature: float = softmax_temperature
        self._model = model
        self._activation_checkpoint: bool = activation_checkpoint
        
        self._future_window_size: int = future_window_size
        self._decay_rate: float = decay_rate
        
        weights = torch.tensor([decay_rate**i for i in range(future_window_size)])
        self.register_buffer('positional_decay_weights', weights)

    def forward(
        self,
        lengths: torch.Tensor,
        output_embeddings: torch.Tensor,   
        supervision_ids: torch.Tensor,    
        supervision_embeddings: torch.Tensor,
        supervision_weights: torch.Tensor,  
        negatives_sampler: NegativesSampler,
        **kwargs,
    ):
        B, N, D = output_embeddings.shape
        W = self._future_window_size
        
        k_indices = torch.arange(1, W + 1, device=output_embeddings.device)

        t_indices = torch.arange(N, device=output_embeddings.device).view(1, N, 1).expand(B, -1, -1)

        future_indices = t_indices + k_indices
        
        padded_ids = F.pad(supervision_ids, (0, W), "constant", 0)
        padded_weights = F.pad(supervision_weights, (0, W), "constant", 0)

        padded_embeddings = F.pad(supervision_embeddings, (0, 0, 0, W), "constant", 0)

        future_supervision_ids = torch.gather(padded_ids, 1, future_indices.view(B, -1)).view(B, N, W)
        future_supervision_weights = torch.gather(padded_weights, 1, future_indices.view(B, -1)).view(B, N, W)
        
        future_supervision_embeddings = torch.gather(
            padded_embeddings, 1, future_indices.view(B, N*W, 1).expand(-1, -1, D)
        ).view(B, N, W, D)

        query_embeddings = output_embeddings.unsqueeze(2).expand(B, N, W, D)

        query_embeddings = query_embeddings.reshape(-1, D)
        positive_ids = future_supervision_ids.reshape(-1)
        positive_embeddings = future_supervision_embeddings.reshape(-1, D)
        per_task_weights = future_supervision_weights.reshape(-1)

        sampled_ids, sampled_negative_embeddings = negatives_sampler(
            positive_ids=positive_ids,
            num_to_sample=self._num_to_sample,
        )
        
        normalized_positive_embeddings = negatives_sampler.normalize_embeddings(positive_embeddings)
        positive_logits, aux_losses = self._model.similarity_fn(
            query_embeddings=query_embeddings,
            item_ids=positive_ids.unsqueeze(1),
            item_embeddings=normalized_positive_embeddings.unsqueeze(1),
            **kwargs,
        )
        sampled_negatives_logits, _ = self._model.similarity_fn(
            query_embeddings=query_embeddings,
            item_ids=sampled_ids,
            item_embeddings=sampled_negative_embeddings,
            **kwargs,
        )
        all_logits = torch.cat([positive_logits, sampled_negatives_logits], dim=1)
        all_logits = all_logits / self._softmax_temperature
        accidental_hits = (positive_ids.unsqueeze(1) == sampled_ids)
        all_logits[:, 1:] = all_logits[:, 1:].masked_fill(accidental_hits, -1e9)
        per_task_loss = -F.log_softmax(all_logits, dim=1)[:, 0]
        decay_weights = self.positional_decay_weights.view(1, 1, W).expand(B, N, W).reshape(-1)
        final_weights = per_task_weights * decay_weights
        weighted_loss = (per_task_loss * final_weights).sum()
        total_weight = final_weights.sum().clamp(min=1e-9)
        final_loss = weighted_loss / total_weight
        
        return final_loss, aux_losses
