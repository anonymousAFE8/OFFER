import abc
import math
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn.functional as F

from generative_recommenders.research.modeling.sequential.embedding_modules import (
    EmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
)
from generative_recommenders.research.modeling.sequential.utils import (
    get_current_embeddings,
)
from generative_recommenders.research.modeling.similarity_module import (
    SequentialEncoderWithLearnedSimilarityModule,
)
from generative_recommenders.research.rails.similarities.module import SimilarityModule

from generative_recommenders.research.modeling.sequential.prediction import BCEPrediction, CEPrediction


TIMESTAMPS_KEY = "timestamps"


class RelativeAttentionBiasModule(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        pass


class RelativePositionalBias(RelativeAttentionBiasModule):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        del all_timestamps
        n: int = self._max_seq_len
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]


class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
        exp_conf_dict: dict,
    ) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len if exp_conf_dict.get("use_user_id", False) == False else max_seq_len - 1
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )
        self.exp_conf_dict = exp_conf_dict

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1 : N]], dim=1
        ) if self.exp_conf_dict.get("use_action", False) == False else torch.cat(
            [all_timestamps, all_timestamps[:, N//2 - 1 : N//2]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        if self.exp_conf_dict.get("use_user_id", False) == True:
            rel_pos_bias = F.pad(rel_pos_bias, pad=(1, 0, 1, 0), mode='constant', value=0)
            rel_ts_bias = F.pad(rel_ts_bias, pad=(1, 0, 1, 0), mode='constant', value=0)
        return rel_pos_bias, rel_ts_bias

class SequentialTransductionUnitJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        concat_ua: bool = False,
        epsilon: float = 1e-6,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = (
            relative_attention_bias_module
        )
        self._normalization: str = normalization
        self._linear_config: str = linear_config
        if self._linear_config == "uvqk":
            self._uvqk: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty(
                    (
                        embedding_dim,
                        linear_hidden_dim * 4 * num_heads
                        + attention_dim * num_heads * 2,
                    )   # 128， 64*2*2+64*2*2。128，512
                ).normal_(mean=0, std=0.02),
            )
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
        self._linear_activation: str = linear_activation
        self._concat_ua: bool = concat_ua
        self._eps: float = epsilon
        
        ffn_multiply = 4
        ffn_single_stage = True
        self._mffn = MultistageFeedforwardNeuralNetwork(
            ams_output_size = linear_hidden_dim * num_heads * 3,
            input_size = embedding_dim,
            hidden_size = int(embedding_dim * ffn_multiply),
            output_size = embedding_dim,
            dropout_ratio = dropout_ratio,
            single_stage = ffn_single_stage,
            epsilon = epsilon
        )
        self._mffn.init()

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads * 3], eps=self._eps
        )

    def forward(  # pyre-ignore [3]
        self,
        x: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        ratings: torch.Tensor,
        item_embs: torch.Tensor,
        invalid_attn_mask: torch.Tensor,
    ):
        B, N, _ = x.size()
        normed_x = self._norm_input(x)
        if self._linear_config == "uvqk":
            batched_mm_output = torch.matmul(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            u, v, q, k = torch.split(
                batched_mm_output,
                [
                    self._linear_dim * self._num_heads * 3,
                    self._linear_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown self._linear_config {self._linear_config}")

        if self._normalization == "rel_bias" or self._normalization == "hstu_rel_bias":
            qk_attn = torch.einsum(
                "bnhd,bmhd->bhnm",
                q.view(B, N, self._num_heads, self._attention_dim),
                k.view(B, N, self._num_heads, self._attention_dim),
            )
            if all_timestamps is not None:
                pos_attn, ts_attn = self._rel_attn_bias(all_timestamps)
                pos_attn = pos_attn * invalid_attn_mask.unsqueeze(0)
                ts_attn = ts_attn * invalid_attn_mask.unsqueeze(0)
            qk_attn = F.silu(qk_attn) / N
            qk_attn = qk_attn * invalid_attn_mask.unsqueeze(0).unsqueeze(0)
            attn_output = torch.einsum(
                    "bhnm,bmhd->bnhd",
                    qk_attn,
                    v.reshape(B, N, self._num_heads, self._linear_dim),
            ).reshape(B, N, self._num_heads * self._linear_dim)
        elif self._normalization == "softmax_rel_bias":
            qk_attn = torch.einsum("bnd,bmd->bnm", q, k)
            if self._rel_attn_bias is not None:
                qk_attn = qk_attn + self._rel_attn_bias(all_timestamps)
            qk_attn = F.softmax(qk_attn / math.sqrt(self._attention_dim), dim=-1)
            qk_attn = qk_attn * invalid_attn_mask
            attn_output = torch.bmm(qk_attn, v)
        else:
            raise ValueError(f"Unknown normalization method {self._normalization}")

        output_pos = torch.einsum("bnm,bmhd->bnhd", pos_attn, v.reshape(B, N, self._num_heads, self._linear_dim))
        output_ts =  torch.einsum("bnm,bmhd->bnhd", ts_attn, v.reshape(B, N, self._num_heads, self._linear_dim))
        output_latent = torch.einsum("bhnm,bmhd->bnhd", qk_attn, v.reshape(B, N, self._num_heads, self._linear_dim))
        
        attn_output = torch.concat(
            [output_pos, output_ts, output_latent],
            dim=-1
        ).reshape(B, N, self._num_heads * self._linear_dim * 3)

        ams_output = u * self._norm_attn_output(attn_output)

        new_outputs = self._mffn(ams_output, x)

        return new_outputs

class MultistageFeedforwardNeuralNetwork(torch.nn.Module) :
    def __init__(
        self, 
        ams_output_size, 
        input_size, 
        hidden_size, 
        output_size, 
        dropout_ratio: float,
        bias: bool = False, 
        single_stage: bool = False,
        epsilon: float = 1e-6,
    ) :
        super(MultistageFeedforwardNeuralNetwork, self).__init__()
        self.lin0 = torch.nn.Linear(ams_output_size, input_size)
        self.is_single_stage = single_stage
        self.dropout_ratio = dropout_ratio
        self.input_size = input_size
        self.eps = epsilon
        if not single_stage :
            self.lin1 = torch.nn.Linear(input_size, hidden_size, bias=bias)
            self.lin2 = torch.nn.Linear(hidden_size, output_size, bias=bias)
            self.lin3 = torch.nn.Linear(input_size, hidden_size, bias=bias)
    
    def forward(self, X, X0) :
        X = (
            self.lin0(
                F.dropout(
                    X,
                    p = self.dropout_ratio,
                    training = self.training
                )
            ) + X0
        )
        if not self.is_single_stage :
            normed_X = F.rms_norm(X, normalized_shape=[self.input_size], eps=self.eps)
            normed_X = F.dropout(
                normed_X,
                p = self.dropout_ratio,
                training = self.training
            )
            X1 = F.silu(self.lin1(normed_X)) * self.lin3(normed_X)
            X = self.lin2(X1) + X
        return X
    
    def init(self) :
        torch.nn.init.xavier_uniform_(self.lin0.weight)
        if not self.is_single_stage :
            torch.nn.init.xavier_uniform_(self.lin1.weight)
            torch.nn.init.xavier_uniform_(self.lin2.weight)
            torch.nn.init.xavier_uniform_(self.lin3.weight)


class HSTUJagged(torch.nn.Module):
    def __init__(
        self,
        modules: List[SequentialTransductionUnitJagged],
        autocast_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()

        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            modules=modules
        )
        self._autocast_dtype: Optional[torch.dtype] = autocast_dtype

    def jagged_forward(
        self,
        x: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        ratings: torch.Tensor,
        item_embs: torch.Tensor,
        invalid_attn_mask: torch.Tensor,
    ):

        with torch.autocast(
            "cuda",
            enabled=self._autocast_dtype is not None,
            dtype=self._autocast_dtype or torch.float16,
        ):
            for i, layer in enumerate(self._attention_layers):
                x = layer(
                    x=x,
                    all_timestamps=all_timestamps,
                    invalid_attn_mask=invalid_attn_mask,
                    ratings = ratings,
                    item_embs = item_embs,
                )

        return x

    def forward(
        self,
        x: torch.Tensor,
        all_timestamps: Optional[torch.Tensor],
        ratings: torch.Tensor,
        item_embs: torch.Tensor,
        invalid_attn_mask: torch.Tensor,
    ):

        y = self.jagged_forward(
            x=x,
            all_timestamps=all_timestamps,
            invalid_attn_mask=invalid_attn_mask,
            ratings=ratings,
            item_embs=item_embs,
        )

        return y

class FUXI(SequentialEncoderWithLearnedSimilarityModule):
    """
    Implements HSTU (Hierarchical Sequential Transduction Unit) in
    Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations,

    Note that this implementation is intended for reproducing experiments in
    the traditional sequential recommender setting (Section 4.1.1), and does
    not yet use optimized kernels discussed in the paper.
    """

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        linear_dim: int,
        attention_dim: int,
        normalization: str,
        linear_config: str,
        linear_activation: str,
        linear_dropout_rate: float,
        attn_dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: SimilarityModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        enable_relative_attention_bias: bool = True,
        concat_ua: bool = False,
        verbose: bool = True,
        exp_conf_dict: dict = {},
        gen_feat_module = None,
        use_timestamps = True,
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        if use_timestamps == False:
            enable_relative_attention_bias = False

        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len
        self._embedding_module: EmbeddingModule = embedding_module
        self._input_features_preproc: InputFeaturesPreprocessorModule = (
            input_features_preproc_module
        )
        self._output_postproc: OutputPostprocessorModule = output_postproc_module
        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._dqk: int = attention_dim
        self._dv: int = linear_dim
        self._linear_activation: str = linear_activation
        self._linear_dropout_rate: float = linear_dropout_rate
        self._attn_dropout_rate: float = attn_dropout_rate
        self._enable_relative_attention_bias: bool = enable_relative_attention_bias
        self._hstu = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(
                    embedding_dim=self._embedding_dim,
                    linear_hidden_dim=linear_dim,
                    attention_dim=attention_dim,
                    normalization=normalization,
                    linear_config=linear_config,
                    linear_activation=linear_activation,
                    num_heads=num_heads,
                    # TODO: change to lambda x.
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=max_sequence_len
                            + max_output_len,  # accounts for next item.
                            num_buckets=128,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
                            exp_conf_dict=exp_conf_dict,
                        )
                        if enable_relative_attention_bias
                        else None
                    ),
                    dropout_ratio=linear_dropout_rate,
                    attn_dropout_ratio=attn_dropout_rate,
                    concat_ua=concat_ua,
                )
                for _ in range(num_blocks)
            ],
            autocast_dtype=None,
        )
        # causal forward, w/ +1 for padding.
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self._max_sequence_length + max_output_len,
                        self._max_sequence_length + max_output_len,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._verbose: bool = verbose
        self.exp_conf_dict = exp_conf_dict
        
        if self.exp_conf_dict["is_recall"] == False:
            if self.exp_conf_dict.get("use_ce_loss", False):
                self.ce_prediction = CEPrediction(self._embedding_dim, exp_conf_dict["num_actions"])
            self.bce_prediction = BCEPrediction(self._embedding_dim)

        self.gen_feat_module = gen_feat_module

    def get_item_embeddings(self, item_ids: torch.Tensor, item_actions=None, user_id=None, item_features=None) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids, item_actions, user_id, item_features)

    def debug_str(self) -> str:
        debug_str = (
            f"HSTU-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        return debug_str

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ):
        device = past_lengths.device
        float_dtype = past_embeddings.dtype
        B, N, _ = past_embeddings.size()    # 256 201 64

        past_lengths, user_embeddings, _ = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )   # 256 201 64
        
        float_dtype = user_embeddings.dtype
        user_embeddings = self._hstu(
            x=user_embeddings,
            all_timestamps=(
                past_payloads[TIMESTAMPS_KEY]
                if TIMESTAMPS_KEY in past_payloads
                else None
            ),
            ratings = past_payloads["ratings"],
            item_embs = past_embeddings,
            invalid_attn_mask=1.0 - self._attn_mask.to(float_dtype),
        )
        
        user_embeddings = self._output_postproc(user_embeddings)
        
        if self.exp_conf_dict.get("use_user_id", False) == True:
            user_embeddings = user_embeddings[:, 1:, :]
        if self.exp_conf_dict.get("use_action", False) == True:
            user_embeddings = user_embeddings[:, ::2, :]
        if self.exp_conf_dict.get("is_recall", False) == False:
            ret = self.getPrediction(user_embeddings)
        else:
            ret = user_embeddings

        return ret

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        encoded_embeddings = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
        return encoded_embeddings

    def _encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) :
        ret = self.generate_user_embeddings(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )  # [B, N, 1]
        encoded_seq_embeddings = ret['bce'] if self.exp_conf_dict["is_recall"] == False else ret
        #如果不间隔行为token，获取预测embedding需要长度减半
        if self.exp_conf_dict.get("use_user_id", False) == True:
            result_lengths = past_lengths - 1
        else:
            result_lengths = past_lengths
        if self.exp_conf_dict.get("use_action", False) == True:
            result_lengths = result_lengths // 2
        current_embeddings = get_current_embeddings(  
            lengths=result_lengths, 
            encoded_embeddings=encoded_seq_embeddings
        )
        
        return current_embeddings #[B, target_num]

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ):
        return self._encode(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

    def getPrediction(self, user_embeddings):
        if self.exp_conf_dict.get("use_ce_loss", False):
            y_ce = self.ce_prediction(user_embeddings)
        else:
            y_ce = None
        y_bce = self.bce_prediction(user_embeddings)
        ret = {
            "ce": y_ce,
            "bce": y_bce,
        }
        return ret





