import logging
import os
import random

import time

from datetime import date, datetime
from typing import Dict, Optional

import gin

import torch
import torch.distributed as dist

from generative_recommenders.research.data.eval import (
    _avg,
    add_to_summary_writer,
    build_eval_output_from_topk_ids,
    eval_metrics_v2_from_tensors,
    get_eval_state,
)

from generative_recommenders.research.data.feature_dataset import get_reco_dataset
from generative_recommenders.research.indexing.utils import get_top_k_module
from generative_recommenders.research.modeling.sequential.autoregressive_losses import (
    BCELoss,
    InBatchNegativesSampler,
    LocalNegativesSampler,
    FeatNegativesSampler,
)
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    EmbeddingModule,
    LocalEmbeddingModule,
    InterEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.encoder_utils import (
    get_sequential_encoder,
)
from generative_recommenders.research.modeling.sequential.features import (
    SequentialFeatures,
    expand_packed_rank_candidates,
    packed_rank_features_from_row,
    seq_features_from_row,
)
from generative_recommenders.research.modeling.sequential.cl4srec import (
    build_cl4srec_views,
    cl4srec_infonce_loss,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.losses.sampled_softmax import (
    SampledSoftmaxLoss,
    WeightedFutureSampledSoftmaxLoss,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
    LayerNormEmbeddingPostprocessor,
)
from generative_recommenders.research.modeling.similarity_utils import (
    get_similarity_function,
)
from generative_recommenders.research.trainer.data_loader import create_data_loader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score, log_loss
from collections import OrderedDict
#from generative_recommenders.research.data.my_dataset import collate_fn
from generative_recommenders.research.modeling.sequential.genfeatures import GenFeatureModule
from generative_recommenders.research.modeling.sequential.optimization import AdamW_with_LAMB_for_Embeddings

def setup_logger(log_dir, log_name="train"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{timestamp}_{log_name}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置最低记录等级为 INFO

    # 清除已有 handler，避免重复输出
    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件输出 handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # 控制台输出 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加 handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # logging.info(f"Logger initialized. Saving logs to {log_path}")
    return logger

def setup(rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()


def load_candidate_topk_map(candidate_topk_path: str):
    if not candidate_topk_path:
        return None
    payload = torch.load(candidate_topk_path, map_location="cpu")
    return {
        int(user_id): payload["top_k_ids"][idx].clone()
        for idx, user_id in enumerate(payload["user_id"].tolist())
    }

def reset_params(model, exp_conf_dict):
    # 需要初始化的 embedding 参数
    init_embedding_list = ["_item_emb", "_feat_emb", "_pos_emb", "_action_emb"]
    init_skip_list = ["_item_pretrain_emb", "_feat_pretrain_emb"]
    #init_skip_list = ["_item_pretrain_emb"]
    if exp_conf_dict.get("load_embs", False):
        init_skip_list.append("down_mlp")
    if exp_conf_dict.get("load_trans", False):
        init_skip_list.append("_hstu")
    
    for name, params in model.named_parameters():
        if any(x in name for x in init_embedding_list):
            torch.nn.init.normal_(params, mean=0.0, std=0.02)
            logging.info(f" Init {name} as normal: {params.data.size()} params")
        elif any(x in name for x in init_skip_list):
            logging.info(f" Skip init for {name}")
        elif len(params.data.size()) >= 2: 
            torch.nn.init.xavier_normal_(params)
            logging.info(f" Init {name} as xavier normal: {params.data.size()} params")
        else:
            torch.nn.init.zeros_(params)
            logging.info(f" Init {name} as zeros: {params.data.size()} params")

def reset_lr(model, learning_rate, exp_conf_dict):
    reduce_params, normal_params = [], []
    lr_decay = exp_conf_dict.get("learning_rate_decay", 1)
    reduce_params_list = ["_embedding_module._item", "_embedding_module._feat", "_embedding_module._user","down_mlp"]
    #reduce_params_list = ["_embedding_module._item", "down_mlp"]
    for name, param in model.named_parameters():
        if lr_decay != 1 and any(x in name for x in reduce_params_list):
            logging.info(f" Reduce learning rate param: {name} to {lr_decay} times")
            reduce_params.append(param)
        else:
            normal_params.append(param)
    return (reduce_params, normal_params), (learning_rate * lr_decay, learning_rate)

@gin.configurable
def get_weighted_loss(
    main_loss: torch.Tensor,
    aux_losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    weighted_loss = main_loss
    for key, weight in weights.items():
        cur_weighted_loss = aux_losses[key] * weight
        weighted_loss = weighted_loss + cur_weighted_loss
    return weighted_loss


def clone_sequential_features(seq_features: SequentialFeatures) -> SequentialFeatures:
    return SequentialFeatures(
        past_lengths=seq_features.past_lengths.clone(),
        past_ids=seq_features.past_ids.clone(),
        past_embeddings=(
            None
            if seq_features.past_embeddings is None
            else seq_features.past_embeddings.clone()
        ),
        past_payloads={
            key: None if value is None else value.clone()
            for key, value in seq_features.past_payloads.items()
        },
        user_id=seq_features.user_id.clone(),
    )


def prepare_encoder_lengths(
    past_lengths: torch.Tensor,
    exp_conf_dict: dict,
) -> torch.Tensor:
    prepared_lengths = past_lengths.clone()
    if exp_conf_dict.get("use_action", False) == True:
        prepared_lengths = prepared_lengths * 2
    if exp_conf_dict.get("use_user_id", False) == True:
        prepared_lengths = prepared_lengths + 1
    return prepared_lengths


def compute_cl4srec_loss(
    model,
    seq_features: SequentialFeatures,
    item_features,
    exp_conf_dict: dict,
    random_seed: int,
) -> torch.Tensor:
    if seq_features.past_ids.size(0) < 2:
        return seq_features.past_ids.new_zeros((), dtype=torch.float32)
    if item_features is not None:
        raise NotImplementedError("CL4SRec auxiliary loss currently assumes use_feat=False")

    view_a, view_b = build_cl4srec_views(
        past_ids=seq_features.past_ids,
        past_lengths=seq_features.past_lengths,
        past_payloads=seq_features.past_payloads,
        crop_ratio=float(exp_conf_dict.get("cl4srec_crop_ratio", 0.6)),
        mask_ratio=float(exp_conf_dict.get("cl4srec_mask_ratio", 0.3)),
        reorder_ratio=float(exp_conf_dict.get("cl4srec_reorder_ratio", 0.2)),
        random_seed=random_seed,
    )

    lengths_a = prepare_encoder_lengths(view_a["past_lengths"], exp_conf_dict)
    lengths_b = prepare_encoder_lengths(view_b["past_lengths"], exp_conf_dict)

    embeddings_a = model.module.get_item_embeddings(
        view_a["past_ids"],
        view_a["past_payloads"]["ratings"],
        seq_features.user_id,
        item_features,
    )
    embeddings_b = model.module.get_item_embeddings(
        view_b["past_ids"],
        view_b["past_payloads"]["ratings"],
        seq_features.user_id,
        item_features,
    )

    repr_a = model.module.encode_hidden(
        past_lengths=lengths_a,
        past_ids=view_a["past_ids"],
        past_embeddings=embeddings_a,
        past_payloads=view_a["past_payloads"],
    )
    repr_b = model.module.encode_hidden(
        past_lengths=lengths_b,
        past_ids=view_b["past_ids"],
        past_embeddings=embeddings_b,
        past_payloads=view_b["past_payloads"],
    )
    return cl4srec_infonce_loss(
        anchor_embeddings=repr_a,
        positive_embeddings=repr_b,
        temperature=float(exp_conf_dict.get("cl4srec_temperature", 0.2)),
    )


def score_packed_rank_candidates(
    model,
    seq_features: SequentialFeatures,
    candidate_ids: torch.Tensor,
    candidate_ratings: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_timestamps: Optional[torch.Tensor],
    item_features,
    exp_conf_dict: dict,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    if item_features is not None:
        raise NotImplementedError("Packed coarse ranking currently assumes use_feat=False")
    num_candidates = candidate_ids.size(1)
    chunk_size = max(1, int(chunk_size or num_candidates))
    score_chunks = []
    for start in range(0, num_candidates, chunk_size):
        end = min(num_candidates, start + chunk_size)
        expanded_seq_features, _expanded_candidate_ratings, _flat_mask = expand_packed_rank_candidates(
            seq_features=seq_features,
            candidate_ids=candidate_ids[:, start:end],
            candidate_ratings=candidate_ratings[:, start:end],
            candidate_mask=candidate_mask[:, start:end],
            target_timestamps=target_timestamps,
        )
        if exp_conf_dict.get("use_action", False) == True:
            expanded_seq_features.past_lengths.mul_(2)
        if exp_conf_dict.get("use_user_id", False) == True:
            expanded_seq_features.past_lengths.add_(1)
        past_embeddings = model.module.get_item_embeddings(
            expanded_seq_features.past_ids,
            expanded_seq_features.past_payloads["ratings"],
            expanded_seq_features.user_id,
            None,
        )
        chunk_scores = model.module.encode(
            past_lengths=expanded_seq_features.past_lengths,
            past_ids=expanded_seq_features.past_ids,
            past_embeddings=past_embeddings,
            past_payloads=expanded_seq_features.past_payloads,
        )
        score_chunks.append(chunk_scores.view(candidate_ids.size(0), end - start))
    return torch.cat(score_chunks, dim=1)


def get_packed_rank_loss(
    candidate_scores: torch.Tensor,
    candidate_ratings: torch.Tensor,
    candidate_mask: torch.Tensor,
    exp_conf_dict: dict,
) -> torch.Tensor:
    if exp_conf_dict.get("ratings_threshold") is not None:
        candidate_labels = candidate_ratings > exp_conf_dict["ratings_threshold"]
    else:
        candidate_labels = candidate_ratings
    candidate_labels = candidate_labels.float()
    bce = torch.nn.BCELoss(reduction="none")(candidate_scores, candidate_labels)
    return bce[candidate_mask].mean()


def eval_packed_rank(
    model,
    device,
    eval_data_loader,
    rank,
    world_size,
    epoch,
    exp_conf_dict,
):
    eval_dict_all = None
    export_rank_path = exp_conf_dict.get("export_rank_path", "")
    export_topk_k = int(exp_conf_dict.get("export_topk_k", 1000))
    export_payload = []
    eval_start_time = time.time()
    model.eval()
    for row in iter(eval_data_loader):
        (
            seq_features,
            target_ids,
            target_ratings,
            candidate_ids,
            candidate_ratings,
            candidate_mask,
            target_timestamps,
        ) = packed_rank_features_from_row(
            row=row,
            device=device,
            max_output_length=exp_conf_dict["rank_num"],
        )
        candidate_scores = score_packed_rank_candidates(
            model=model,
            seq_features=seq_features,
            candidate_ids=candidate_ids,
            candidate_ratings=candidate_ratings,
            candidate_mask=candidate_mask,
            target_timestamps=target_timestamps,
            item_features=None,
            exp_conf_dict=exp_conf_dict,
            chunk_size=exp_conf_dict.get("packed_eval_chunk_size", 100),
        )
        masked_scores = candidate_scores.masked_fill(~candidate_mask, float("-inf"))
        sorted_scores, sorted_indices = torch.sort(masked_scores, dim=1, descending=True)
        sorted_ids = torch.gather(candidate_ids, dim=1, index=sorted_indices)
        eval_dict = build_eval_output_from_topk_ids(
            eval_top_k_ids=sorted_ids,
            eval_top_k_prs=sorted_scores,
            target_ids=target_ids,
            device=device,
            target_ratings=target_ratings,
            return_topk=bool(export_rank_path),
            max_k=max(export_topk_k, candidate_ids.size(1)),
        )

        if eval_dict_all is None:
            eval_dict_all = {}
            for key, value in eval_dict.items():
                if key in {"top_k_ids", "top_k_scores"}:
                    continue
                eval_dict_all[key] = [value]
        else:
            for key, value in eval_dict.items():
                if key in {"top_k_ids", "top_k_scores"}:
                    continue
                eval_dict_all[key].append(value)

        if export_rank_path:
            export_payload.append(
                {
                    "user_id": row["user_id"].detach().cpu(),
                    "target_id": target_ids.detach().cpu().squeeze(1),
                    "target_rating": target_ratings.detach().cpu().squeeze(1),
                    "top_k_ids": eval_dict["top_k_ids"][:, :export_topk_k].detach().cpu(),
                    "top_k_scores": eval_dict["top_k_scores"][:, :export_topk_k].detach().cpu(),
                }
            )

    assert eval_dict_all is not None
    for key, values in eval_dict_all.items():
        eval_dict_all[key] = torch.cat(values, dim=-1)

    metrics = {
        "ndcg@10": _avg(eval_dict_all["ndcg@10"], world_size=world_size),
        "ndcg@50": _avg(eval_dict_all["ndcg@50"], world_size=world_size),
        "hr@10": _avg(eval_dict_all["hr@10"], world_size=world_size),
        "hr@50": _avg(eval_dict_all["hr@50"], world_size=world_size),
        "mrr": _avg(eval_dict_all["mrr"], world_size=world_size),
    }
    logging.info(
        f"rank {rank}: packed eval @ epoch {epoch} in {time.time() - eval_start_time:.2f}s: "
        f"NDCG@10 {metrics['ndcg@10']:.4f}, NDCG@50 {metrics['ndcg@50']:.4f}, "
        f"HR@10 {metrics['hr@10']:.4f}, HR@50 {metrics['hr@50']:.4f}, MRR {metrics['mrr']:.4f}"
    )

    if export_rank_path and rank == 0:
        os.makedirs(os.path.dirname(export_rank_path) or ".", exist_ok=True)
        torch.save(
            {
                "user_id": torch.cat([x["user_id"] for x in export_payload], dim=0),
                "target_id": torch.cat([x["target_id"] for x in export_payload], dim=0),
                "target_rating": torch.cat([x["target_rating"] for x in export_payload], dim=0),
                "top_k_ids": torch.cat([x["top_k_ids"] for x in export_payload], dim=0),
                "top_k_scores": torch.cat([x["top_k_scores"] for x in export_payload], dim=0),
            },
            export_rank_path,
        )
        logging.info(f"Saved packed rank top-{export_topk_k} candidates to {export_rank_path}")

    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


@gin.configurable
def train_fn(
    rank: int,
    world_size: int,
    master_port: int,
    dataset_name: str = "ml-20m",
    max_sequence_length: int = 200,
    local_batch_size: int = 128,
    eval_batch_size: int = 128,
    eval_user_max_batch_size: Optional[int] = None,
    main_module: str = "SASRec",
    main_module_bf16: bool = False,
    dropout_rate: float = 0.2,
    user_embedding_norm: str = "l2_norm",
    sampling_strategy: str = "in-batch",
    loss_module: str = "SampledSoftmaxLoss",
    loss_weights: Optional[Dict[str, float]] = {},
    num_negatives: int = 1,
    loss_activation_checkpoint: bool = False,
    item_l2_norm: bool = False,
    temperature: float = 0.05,
    num_epochs: int = 101,
    learning_rate: float = 1e-3,
    num_warmup_steps: int = 0,
    weight_decay: float = 1e-3,
    top_k_method: str = "MIPSBruteForceTopK",
    eval_interval: int = 100,
    full_eval_every_n: int = 1,
    save_ckpt_every_n: int = 1000,
    partial_eval_num_iters: int = 32,
    item_embedding_dim: int = 240,
    feat_embedding_dim: int = 64,
    interaction_module_type: str = "",
    l2_norm_eps: float = 1e-6,
    enable_tf32: bool = False,
    random_seed: int = 42,
    exp_conf_dict: dict = {},
) -> None:
    # to enable more deterministic results.
    random.seed(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    logging.info(f"cuda.matmul.allow_tf32: {enable_tf32}")
    logging.info(f"cudnn.allow_tf32: {enable_tf32}")
    logging.info(f"Training model on rank {rank}.")
    setup(rank, world_size, master_port)
    
    use_timestamps = False if dataset_name == "QBArticle" else True

    dataset = get_reco_dataset(
        dataset_name=dataset_name,
        max_sequence_length=max_sequence_length,
        chronological=True,
        exp_conf_dict = exp_conf_dict,
    )

    train_data_sampler, train_data_loader = create_data_loader(
        dataset=dataset.train_dataset,
        batch_size=local_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        drop_last=world_size > 1,
    )
 
    eval_data_sampler, eval_data_loader = create_data_loader(
        dataset=dataset.eval_dataset,
        batch_size=eval_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,  # needed for partial eval
        drop_last=world_size > 1,
    )

    model_debug_str = main_module
    common_args = {
        "num_items": dataset.max_item_id,
        "num_users": dataset.max_user_id,
        "item_embedding_dim": item_embedding_dim,
        "feat_embedding_dim": feat_embedding_dim,
        "exp_conf_dict": exp_conf_dict,
        "feat_info_dict": dataset.feat_info_dict if exp_conf_dict.get("use_feat") == True else None,
        "feat_dim_dict": dataset.feat_dim_dict if exp_conf_dict.get("use_feat") == True else None,
    }
    
    if exp_conf_dict.get("use_action", False) == False or exp_conf_dict.get("add_action", False) == True:
        EmbeddingClass = LocalEmbeddingModule
    elif exp_conf_dict.get("use_action", False) == True:
        EmbeddingClass = InterEmbeddingModule
    embedding_module: EmbeddingModule = EmbeddingClass(**common_args)
    model_debug_str += f"-{embedding_module.debug_str()}"
    
    if dataset_name == "KuaiRandPure":
        feat_names = ["video_type", "upload_dt", "upload_type", "music_type"]
        feat_pre_info_dict = "data/KuaiRandPure/feat_inter_info.json"
        num_masked_features = 2
        lambda_val = 0.5
    if dataset_name == "AliEC":
        feat_names = ["cate_id", "campaign_id", "customer", "brand", "price"]
        feat_pre_info_dict = "data/AliEC/feat_inter_info.json"
        num_masked_features = 2
        lambda_val = 0.3
    if dataset_name == "Yelp":
        feat_names = ["city", "state", "postal_code", "stars", "review_count", "is_open"]
        feat_pre_info_dict = "data/Yelp/feat_inter_info.json"
        num_masked_features = 3
        lambda_val = 0.5

    if exp_conf_dict.get("use_gen_feat", False):
        gen_feat_module = GenFeatureModule(
            item_emb = embedding_module._item_emb,
            feat_emb = embedding_module._feat_emb,
            feat_names = feat_names,
            hidden_dim = 32,
            proj_dim = 64,
            feat_pre_info_dict = feat_pre_info_dict,
            loss_type = "contrastive",
            temperature = 0.1,
            mask_mode = "soft",       
            dropout_p = 0.5,
            num_masked_features = num_masked_features,
            lambda_val = lambda_val,        
        )
    else:
        gen_feat_module = None

    assert (
        user_embedding_norm == "l2_norm" or user_embedding_norm == "layer_norm"
    ), f"Not implemented for {user_embedding_norm}"
    output_postproc_module = (
        L2NormEmbeddingPostprocessor(
            embedding_dim=item_embedding_dim,
            eps=1e-6,
        )
        if user_embedding_norm == "l2_norm"
        else LayerNormEmbeddingPostprocessor(
            embedding_dim=item_embedding_dim,
            eps=1e-6,
        )
    )
    input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len= ( dataset.max_sequence_length + exp_conf_dict["rank_num"] if exp_conf_dict.get("use_action", False) == False
                else ( dataset.max_sequence_length + exp_conf_dict["rank_num"] )*2 ),
        embedding_dim=item_embedding_dim,
        dropout_rate=dropout_rate,
        exp_conf_dict = exp_conf_dict,
    )

    
    interaction_module, _ = get_similarity_function(
        module_type=interaction_module_type,
        query_embedding_dim=item_embedding_dim,
        item_embedding_dim=item_embedding_dim,
    )

    model = get_sequential_encoder(
        module_type=main_module,
        max_sequence_length=( dataset.max_sequence_length + int(exp_conf_dict.get("use_user_id", False)) 
                              if exp_conf_dict.get("use_action", False) == False
                              else ( dataset.max_sequence_length * 2 + int(exp_conf_dict.get("use_user_id", False)) )),
        max_output_length = exp_conf_dict["rank_num"] if exp_conf_dict.get("use_action", False) == False else exp_conf_dict["rank_num"] * 2,
        embedding_module=embedding_module,
        interaction_module=interaction_module,
        input_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        verbose=True,
        exp_conf_dict = exp_conf_dict,
        gen_feat_module = gen_feat_module,
        use_timestamps = use_timestamps,
    )
    model_debug_str = model.debug_str()

    # 载入预训练模型
    if exp_conf_dict.get("load_trans", False):
        full_model_weights = torch.load(exp_conf_dict["model_dir"])
        transformer_weights_filtered = OrderedDict()
        key_name = '_hstu.'
        for key, value in full_model_weights.items():
            if key_name in key: 
                transformer_weights_filtered[ key[len(key_name):] ] = value
                logging.info(f"Load param {key}")
        model._hstu.load_state_dict(transformer_weights_filtered)
    # 统一初始化模型参数
    reset_params(model, exp_conf_dict)
    full_model_load_path = exp_conf_dict.get("load_full_model_path", "")
    if full_model_load_path:
        full_model_weights = torch.load(full_model_load_path, map_location="cpu")
        load_result = model.load_state_dict(full_model_weights, strict=False)
        logging.info(
            f"Loaded full model state from {full_model_load_path} "
            f"(missing={list(load_result.missing_keys)}, unexpected={list(load_result.unexpected_keys)})"
        )
    
    if exp_conf_dict["is_recall"] == True:
        if exp_conf_dict.get("use_future_loss", False):
            ar_loss = WeightedFutureSampledSoftmaxLoss(
                        num_to_sample=num_negatives,
                        softmax_temperature=temperature,
                        model=model,
                        activation_checkpoint=loss_activation_checkpoint,
                        future_window_size = 4,       
                        decay_rate = 0.9,  
                    )
        else:
            ar_loss = SampledSoftmaxLoss(
                        num_to_sample=num_negatives,
                        softmax_temperature=temperature,
                        model=model,
                        activation_checkpoint=loss_activation_checkpoint,
                    )
        if exp_conf_dict.get("use_feat", False) == False:
            negatives_sampler = LocalNegativesSampler(
                num_items=dataset.max_item_id,
                item_emb=model._embedding_module._item_emb,
                all_item_ids=dataset.all_item_ids,
                l2_norm=item_l2_norm,
                l2_norm_eps=l2_norm_eps,
            )
        else:
            negatives_sampler = FeatNegativesSampler(
                num_items=dataset.max_item_id,
                all_item_ids=dataset.all_item_ids,
                l2_norm=item_l2_norm,
                l2_norm_eps=l2_norm_eps,
                get_feat=dataset.train_dataset.look_for_feat_fast_dict,
                feat_emb=model._embedding_module.get_item_embeddings,
            )

    result_dir = exp_conf_dict.get("result_dir", f"./logs/{dataset_name}")
    logger = setup_logger(
        result_dir,
        f"epoch{num_epochs}_bs{local_batch_size}" + f"dim{item_embedding_dim}_{model_debug_str}",
    )

    # Creates model and moves it to GPU with id rank
    device = rank
    if main_module_bf16:
        model = model.to(torch.bfloat16)
    model = model.to(device)
    model = DDP(model, device_ids=[rank], broadcast_buffers=False)
    
    params_group, lr_group = reset_lr(model, learning_rate, exp_conf_dict)
    
    param_groups = [
    {
        'params': params_group[0],
        'lr': lr_group[0],
        'apply_lamb': True, 
        'name': 'pretrain_params'
    },
    {
        'params': params_group[1],
        'lr': lr_group[1],
        'apply_lamb': False, 
        'name': 'model_params'
    }
    ]
    
    if exp_conf_dict.get("use_lamb", False) == True:
        opt = AdamW_with_LAMB_for_Embeddings(
            param_groups,
            lr=learning_rate,
            betas =  (0.9, 0.98),
            weight_decay=weight_decay,
        )
    else:
        opt = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.98),
            weight_decay=weight_decay,
        )

    date_str = date.today().strftime("%Y-%m-%d")
    model_subfolder = f"{dataset_name}-l{max_sequence_length}"
    model_desc = (
        f"{model_subfolder}"
        + f"/{model_debug_str}"
        + f"{f'-ddp{world_size}' if world_size > 1 else ''}-b{local_batch_size}-lr{learning_rate}-wu{num_warmup_steps}-wd{weight_decay}{'' if enable_tf32 else '-notf32'}-{date_str}"
    )
    if full_eval_every_n > 1:
        model_desc += f"-fe{full_eval_every_n}"
    # creates subfolders.
    os.makedirs(f"./exps/{model_subfolder}", exist_ok=True)
    os.makedirs(f"./ckpts/{model_subfolder}", exist_ok=True)
    log_dir = f"./exps/{model_desc}"
    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)
        logging.info(f"Rank {rank}: writing logs to {log_dir}")
    else:
        writer = None
        logging.info(f"Rank {rank}: disabling summary writer")

    last_training_time = time.time()
    torch.autograd.set_detect_anomaly(True)

    batch_id = 0
    epoch = 0
    best_epoch, best_auc = 0, 0
    best_logloss = float("inf")
    best_rank_metrics = {
        "ndcg@10": 0.0,
        "ndcg@50": 0.0,
        "hr@10": 0.0,
        "hr@50": 0.0,
        "mrr": 0.0,
    }
    Loss_list = []
    initial_lrs = [pg['lr'] for pg in opt.param_groups]
    eval_only = bool(exp_conf_dict.get("eval_only", False))
    for epoch in range(num_epochs):
        if train_data_sampler is not None:
            train_data_sampler.set_epoch(epoch)
        if eval_data_sampler is not None:
            eval_data_sampler.set_epoch(epoch)
        if eval_only:
            if rank == 0:
                logging.info("eval_only=True: skipping training and running evaluation/export only.")
        else:
            model.train()
            for row in iter(train_data_loader):
                if exp_conf_dict.get("use_feat"):
                    item_features = row["item_feat"]
                    for k, v in item_features.items():
                        item_features[k] = v.to(device)
                else:
                    item_features = None
                packed_rank_payload = None
                if exp_conf_dict.get("packed_ranking", False):
                    (
                        seq_features,
                        _target_ids,
                        _target_ratings,
                        candidate_ids,
                        candidate_ratings,
                        candidate_mask,
                        target_timestamps,
                    ) = packed_rank_features_from_row(
                        row,
                        device=device,
                        max_output_length=exp_conf_dict["rank_num"],
                    )
                    packed_rank_payload = (
                        candidate_ids,
                        candidate_ratings,
                        candidate_mask,
                        target_timestamps,
                    )
                else:
                    seq_features, target_ids, target_ratings = seq_features_from_row(
                        row,
                        device=device,
                        max_output_length=exp_conf_dict["rank_num"],
                    )
                    seq_features.past_ids.scatter_(
                        dim=1,
                        index=seq_features.past_lengths.view(-1, 1),
                        src=target_ids.view(-1, 1),
                    )
                    seq_features.past_payloads["ratings"].scatter_(
                        dim=1,
                        index=seq_features.past_lengths.view(-1, 1),
                        src=target_ratings.view(-1, 1),
                    )
                    seq_features.past_lengths.add_(1)
                    if exp_conf_dict.get("use_action", False) == True:
                       seq_features.past_lengths.mul_(2)
                    if exp_conf_dict.get("use_user_id", False) == True:
                       seq_features.past_lengths.add_(1)

                contrastive_seq_features = None
                if exp_conf_dict.get("use_cl4srec", False):
                    contrastive_seq_features = clone_sequential_features(seq_features)
                
                opt.zero_grad()

                if exp_conf_dict.get("use_gen_feat",False):
                    gen_feat_loss = model.module.gen_feat_module.training_step(seq_features.past_ids, item_features)
                else:
                    gen_feat_loss = None

                if exp_conf_dict.get("use_cl4srec", False):
                    cl4srec_loss = compute_cl4srec_loss(
                        model=model,
                        seq_features=contrastive_seq_features,
                        item_features=item_features,
                        exp_conf_dict=exp_conf_dict,
                        random_seed=random_seed + batch_id,
                    )
                else:
                    cl4srec_loss = None

                if exp_conf_dict["is_recall"] == True:
                    input_embeddings = model.module.get_item_embeddings(seq_features.past_ids,seq_features.past_payloads["ratings"], seq_features.user_id, item_features)                            
                    ret = model(
                        past_lengths=seq_features.past_lengths,
                        past_ids=seq_features.past_ids,
                        past_embeddings=input_embeddings,
                        past_payloads=seq_features.past_payloads,
                    )
                    supervision_ids = seq_features.past_ids
                    ratings = seq_features.past_payloads['ratings']
                    ar_loss = ar_loss.to(device)
                    negatives_sampler = negatives_sampler.to(device)
                    ar_mask = supervision_ids[:, 1:] != 0
                    if exp_conf_dict.get("use_user_id", False) == True:
                        input_embeddings = input_embeddings[:, 1:, :]
                    
                    if exp_conf_dict.get("use_loss_weight", False) == True:
                        if exp_conf_dict.get("ratings_threshold") is not None:
                            temp_mask = torch.where(ratings[:,1:] > exp_conf_dict["ratings_threshold"], 1.0, 0.0)
                        else:
                            temp_mask = ratings[:,1:]
                        ar_mask = ar_mask.float() * temp_mask
                    loss, aux_losses = ar_loss(
                        lengths=seq_features.past_lengths,  # [B],
                        output_embeddings=ret[:, :-1, :],  # [B, N-1, D]
                        supervision_ids=supervision_ids[:, 1:],  # [B, N-1]
                        supervision_embeddings=input_embeddings[:, 1:, :],  # [B, N - 1, D]
                        supervision_weights=ar_mask.float(),
                        negatives_sampler=negatives_sampler,
                        **seq_features.past_payloads,
                    )  # [B, N]
                    loss = get_weighted_loss(loss, aux_losses, weights=loss_weights or {})

                    if gen_feat_loss is not None:
                        loss = loss + 0.2 * gen_feat_loss
                    if cl4srec_loss is not None:
                        loss = loss + float(exp_conf_dict.get("cl4srec_weight", 0.1)) * cl4srec_loss
                elif exp_conf_dict.get("packed_ranking", False):
                    candidate_ids, candidate_ratings, candidate_mask, target_timestamps = packed_rank_payload
                    candidate_scores = score_packed_rank_candidates(
                        model=model,
                        seq_features=seq_features,
                        candidate_ids=candidate_ids,
                        candidate_ratings=candidate_ratings,
                        candidate_mask=candidate_mask,
                        target_timestamps=target_timestamps,
                        item_features=item_features,
                        exp_conf_dict=exp_conf_dict,
                        chunk_size=exp_conf_dict.get("packed_train_chunk_size", candidate_ids.size(1)),
                    )
                    loss = get_packed_rank_loss(
                        candidate_scores=candidate_scores,
                        candidate_ratings=candidate_ratings,
                        candidate_mask=candidate_mask,
                        exp_conf_dict=exp_conf_dict,
                    )
                    if gen_feat_loss is not None:
                        loss = loss + 0.2 * gen_feat_loss
                    if cl4srec_loss is not None:
                        loss = loss + float(exp_conf_dict.get("cl4srec_weight", 0.1)) * cl4srec_loss
                else:
                    input_embeddings = model.module.get_item_embeddings(seq_features.past_ids,seq_features.past_payloads["ratings"], seq_features.user_id, item_features)                            
                    ret = model(
                        past_lengths=seq_features.past_lengths,
                        past_ids=seq_features.past_ids,
                        past_embeddings=input_embeddings,
                        past_payloads=seq_features.past_payloads,
                    )  # {}
                    supervision_ids = seq_features.past_ids
                    ratings = seq_features.past_payloads['ratings']
                    loss = get_rank_loss(ret, ratings, supervision_ids, exp_conf_dict)

                    if gen_feat_loss is not None:
                        loss = loss + 0.2 * gen_feat_loss
                    if cl4srec_loss is not None:
                        loss = loss + float(exp_conf_dict.get("cl4srec_weight", 0.1)) * cl4srec_loss

                """B, N, D = seq_embeddings.shape
                mask_bce_loss = MaskBCELoss(D, [32, 16, 1], ar_mask, seq_embeddings.device)
                loss = mask_bce_loss(seq_embeddings, seq_features.past_payloads['ratings'])"""
                main_loss = loss.detach().clone()
                loss = get_weighted_loss(loss, aux_losses={}, weights=loss_weights or {})

                if rank == 0:
                    assert writer is not None
                    writer.add_scalar("losses/ar_loss", loss, batch_id)
                    writer.add_scalar("losses/main_loss", main_loss, batch_id)
                    if gen_feat_loss is not None:
                        writer.add_scalar("losses/gen_feat_loss", gen_feat_loss, batch_id)
                    if cl4srec_loss is not None:
                        writer.add_scalar("losses/cl4srec_loss", cl4srec_loss, batch_id)

                loss.backward()
                
                # 检测异常点
                # none_grad_params = []
                # for name, param in model.named_parameters():
                #     if param.requires_grad and param.grad is None:
                #         none_grad_params.append(name)
                # print(none_grad_params)
                # exit()

                # Optional linear warmup.
                if batch_id < num_warmup_steps:
                    lr_scalar = min(1.0, float(batch_id + 1) / num_warmup_steps)
                    for i, pg in enumerate(opt.param_groups):
                        pg["lr"] = lr_scalar * initial_lrs[i]
                else:
                    for i, pg in enumerate(opt.param_groups):
                        pg['lr'] = initial_lrs[i]
                
                if (batch_id % eval_interval) == 0:
                    if rank == 0:
                        logging.info(
                            f" rank: {rank}, batch-stat (train): step {batch_id} "
                            f"(epoch {epoch} in {time.time() - last_training_time:.2f}s): {loss:.6f}"
                        )
                        Loss_list.append(round(float(loss.detach().cpu().numpy()),2))
                    last_training_time = time.time()
                    if rank == 0:
                        assert writer is not None
                        writer.add_scalar("loss/train", loss, batch_id)
                        #writer.add_scalar("lr", lr, batch_id)

                opt.step()

                batch_id += 1

        def is_full_eval(epoch: int) -> bool:
            return (epoch % full_eval_every_n) == 0

        eval_start_time = time.time()
        model.eval()

        if exp_conf_dict["is_recall"] == True:
            eval_recall(model, dataset, negatives_sampler, top_k_method, device, 
                main_module_bf16, eval_data_loader, exp_conf_dict["rank_num"],
                eval_user_max_batch_size, partial_eval_num_iters, is_full_eval,
                epoch, world_size, rank, exp_conf_dict)
            continue
        if exp_conf_dict.get("packed_ranking", False):
            current_rank_metrics = eval_packed_rank(
                model=model,
                device=device,
                eval_data_loader=eval_data_loader,
                rank=rank,
                world_size=world_size,
                epoch=epoch,
                exp_conf_dict=exp_conf_dict,
            )
            if current_rank_metrics["ndcg@50"] > best_rank_metrics["ndcg@50"]:
                best_epoch = epoch
                best_rank_metrics = current_rank_metrics
            continue

        y_true_list, y_score_list = [], []
        export_rank_path = exp_conf_dict.get("export_rank_path", "")
        export_rank_payload = []
        for eval_iter, row in enumerate(iter(eval_data_loader)):
            item_features = None
            if exp_conf_dict.get("use_feat", False):
                item_features = row["item_feat"]
                for k, v in item_features.items():
                    item_features[k] = v.to(device)
            seq_features, target_ids, target_ratings = seq_features_from_row(
                row, device=device, max_output_length=exp_conf_dict["rank_num"],
            )
            seq_features.past_ids.scatter_(
                dim=1,
                index=seq_features.past_lengths.view(-1, 1),
                src=target_ids.view(-1, 1),
            )
            seq_features.past_lengths.add_(1)
            if exp_conf_dict.get("use_action", False) == True:
               seq_features.past_lengths.mul_(2)
            if exp_conf_dict.get("use_user_id", False) == True:
               seq_features.past_lengths.add_(1)
              
            past_embeddings = model.module.get_item_embeddings(seq_features.past_ids,seq_features.past_payloads["ratings"], seq_features.user_id, item_features)
            
            y_score = model.module.encode(
                past_lengths=seq_features.past_lengths,
                past_ids=seq_features.past_ids,
                past_embeddings=past_embeddings,
                past_payloads=seq_features.past_payloads,
            )
            # 如果有阈值划分则将多分类变成二分类
            if exp_conf_dict.get("ratings_threshold") is not None:
                y_true = target_ratings > exp_conf_dict["ratings_threshold"]
            else:
                y_true = target_ratings
            if export_rank_path:
                export_rank_payload.append(
                    {
                        "user_id": row["user_id"].detach().cpu(),
                        "target_id": target_ids.detach().cpu().squeeze(1),
                        "target_rating": target_ratings.detach().cpu().squeeze(1),
                        "y_true": y_true.detach().cpu().view(-1),
                        "y_score": y_score.detach().cpu().view(-1),
                    }
                )
            y_true = y_true.long()
            y_true = y_true.view(-1).detach().cpu().numpy()
            y_score = y_score.view(-1).detach().cpu().numpy()
            y_true_list.extend(y_true)
            y_score_list.extend(y_score)

        auc_value = roc_auc_score(y_true_list, y_score_list)
        log_loss_value = log_loss(y_true_list, y_score_list)
        logging.info(f"eval epoch {epoch} AUC:{auc_value:.4f} LogLoss:{log_loss_value:.4f}")
        if rank == 0 and export_rank_path:
            os.makedirs(os.path.dirname(export_rank_path) or ".", exist_ok=True)
            export_rank_dict = {}
            for key in ["user_id", "target_id", "target_rating", "y_true", "y_score"]:
                export_rank_dict[key] = torch.cat([batch[key] for batch in export_rank_payload], dim=0)
            torch.save(export_rank_dict, export_rank_path)
            logging.info(f"Saved rank predictions to {export_rank_path}")
        if auc_value > best_auc:
            best_epoch = epoch
            best_auc, best_logloss = auc_value, log_loss_value
    
    if exp_conf_dict["is_recall"] == False:
        logging.info(f"Loss_List={Loss_list}")
        if exp_conf_dict.get("packed_ranking", False):
            logging.info(
                f"best result at epoch {best_epoch} "
                f"NDCG@10:{best_rank_metrics['ndcg@10']:.4f} "
                f"NDCG@50:{best_rank_metrics['ndcg@50']:.4f} "
                f"HR@10:{best_rank_metrics['hr@10']:.4f} "
                f"HR@50:{best_rank_metrics['hr@50']:.4f} "
                f"MRR:{best_rank_metrics['mrr']:.4f}"
            )
        else:
            logging.info(f"best result at epoch {best_epoch} AUC:{best_auc:.4f} LogLoss:{best_logloss:.4f}")
    
    if exp_conf_dict["save_model"] == True:
        torch.save(model.module.state_dict(), exp_conf_dict["model_dir"])
        if rank == 0:
            logging.info(f"Save model in {exp_conf_dict['model_dir']}")

    cleanup()


def get_rank_loss(ret, ratings, past_ids, exp_conf_dict):
    ar_mask = past_ids != 0
    ar_mask = ar_mask.view(-1)
    loss = 0.0

    loss_fn = {
        'ce': torch.nn.CrossEntropyLoss(reduction='none'),
        'bce': torch.nn.BCELoss(reduction='none') 
    }    
    
    if exp_conf_dict.get("use_ce_loss", False):
        pre_ce = ret['ce']
        pre_ce = pre_ce.view(-1, exp_conf_dict["num_actions"])
        ratings_ce = ratings.long()     
        ratings_ce = ratings_ce.view(-1)
        loss_ce = loss_fn['ce'](pre_ce, ratings_ce)
        loss_ce = loss_ce[ar_mask].mean()
        loss += loss_ce
    
    # 必须使用bce损失作为rank_loss之一
    pre_bce = ret['bce']
    pre_bce = pre_bce.view(-1, 1)
    if exp_conf_dict.get("ratings_threshold") is not None:
       ratings_bce = ratings > exp_conf_dict["ratings_threshold"]
    else:
       ratings_bce = ratings
    ratings_bce = ratings_bce.float().view(-1, 1)
    loss_bce = loss_fn['bce'](pre_bce, ratings_bce)
    loss_bce = loss_bce[ar_mask].mean()
    loss += loss_bce

    return loss

def eval_recall(model, dataset, negatives_sampler, top_k_method, device, 
                main_module_bf16, eval_data_loader, gr_output_length,
                eval_user_max_batch_size, partial_eval_num_iters, is_full_eval,
                epoch, world_size, rank, exp_conf_dict):
        eval_dict_all = None
        export_topk_path = exp_conf_dict.get("export_topk_path", "")
        export_topk_k = int(exp_conf_dict.get("export_topk_k", 1000))
        candidate_topk_path = exp_conf_dict.get("candidate_topk_path", "")
        candidate_topk_map = load_candidate_topk_map(candidate_topk_path) if candidate_topk_path else None
        if candidate_topk_map is not None and rank == 0:
            logging.info(f"Loaded restricted candidate pools from {candidate_topk_path} for {len(candidate_topk_map)} users")
        export_payload = []
        eval_start_time = time.time()
        model.eval()
        eval_state = get_eval_state(
            model=model.module,
            all_item_ids=dataset.all_item_ids,
            negatives_sampler=negatives_sampler,
            top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
                top_k_method=top_k_method,
                model=model.module,
                item_embeddings=item_embeddings,
                item_ids=item_ids,
            ),
            device=device,
            float_dtype=torch.bfloat16 if main_module_bf16 else None,
            exp_conf_dict = exp_conf_dict,
            get_feat=dataset.train_dataset.look_for_feat_fast_dict,
            feat_emb=model.module._embedding_module.get_item_embeddings,
        )
        for eval_iter, row in enumerate(iter(eval_data_loader)):
            seq_features, target_ids, target_ratings = seq_features_from_row(
                row, device=device, max_output_length=gr_output_length,
            )

            if exp_conf_dict.get("use_action", False) == True:
               seq_features.past_lengths.mul_(2)
            if exp_conf_dict.get("use_user_id", False) == True:
               seq_features.past_lengths.add_(1)

            candidate_topk_ids = None
            if candidate_topk_map is not None:
                user_ids = row["user_id"].detach().cpu().tolist()
                missing_user_ids = [user_id for user_id in user_ids if int(user_id) not in candidate_topk_map]
                if missing_user_ids:
                    raise KeyError(f"Missing restricted candidates for users: {missing_user_ids[:5]}")
                candidate_topk_ids = torch.stack(
                    [candidate_topk_map[int(user_id)] for user_id in user_ids],
                    dim=0,
                ).to(device)

            eval_dict = eval_metrics_v2_from_tensors(
                eval_state=eval_state,
                model=model.module,
                row=row,
                seq_features=seq_features,
                target_ids=target_ids,
                target_ratings=target_ratings,
                user_max_batch_size=eval_user_max_batch_size,
                dtype=torch.bfloat16 if main_module_bf16 else None,
                exp_conf_dict=exp_conf_dict,
                return_topk=bool(export_topk_path),
                candidate_topk_ids=candidate_topk_ids,
            )

            if eval_dict_all is None:
                eval_dict_all = {}
                for k, v in eval_dict.items():
                    if k in {"top_k_ids", "top_k_scores"}:
                        continue
                    eval_dict_all[k] = []

            for k, v in eval_dict.items():
                if k in {"top_k_ids", "top_k_scores"}:
                    continue
                eval_dict_all[k] = eval_dict_all[k] + [v]

            if export_topk_path:
                export_payload.append(
                    {
                        "user_id": row["user_id"].detach().cpu(),
                        "target_id": target_ids.detach().cpu().squeeze(1),
                        "target_rating": target_ratings.detach().cpu().squeeze(1),
                        "top_k_ids": eval_dict["top_k_ids"][:, :export_topk_k].detach().cpu(),
                        "top_k_scores": eval_dict["top_k_scores"][:, :export_topk_k].detach().cpu(),
                    }
                )
            del eval_dict

            if (eval_iter + 1 >= partial_eval_num_iters) and (not is_full_eval(epoch)):
                logging.info(
                    f"Truncating epoch {epoch} eval to {eval_iter + 1} iters to save cost.."
                )
                break

        assert eval_dict_all is not None
        for k, v in eval_dict_all.items():
            eval_dict_all[k] = torch.cat(v, dim=-1)

        ndcg_10 = _avg(eval_dict_all["ndcg@10"], world_size=world_size)
        ndcg_50 = _avg(eval_dict_all["ndcg@50"], world_size=world_size)
        hr_10 = _avg(eval_dict_all["hr@10"], world_size=world_size)
        hr_50 = _avg(eval_dict_all["hr@50"], world_size=world_size)
        mrr = _avg(eval_dict_all["mrr"], world_size=world_size)

        logging.info(
            f"rank {rank}: eval @ epoch {epoch} in {time.time() - eval_start_time:.2f}s: "
            f"NDCG@10 {ndcg_10:.4f}, NDCG@50 {ndcg_50:.4f}, HR@10 {hr_10:.4f}, HR@50 {hr_50:.4f}, MRR {mrr:.4f}"
        )

        if export_topk_path and rank == 0:
            export_dir = os.path.dirname(export_topk_path) or "."
            os.makedirs(export_dir, exist_ok=True)
            torch.save(
                {
                    "user_id": torch.cat([x["user_id"] for x in export_payload], dim=0),
                    "target_id": torch.cat([x["target_id"] for x in export_payload], dim=0),
                    "target_rating": torch.cat([x["target_rating"] for x in export_payload], dim=0),
                    "top_k_ids": torch.cat([x["top_k_ids"] for x in export_payload], dim=0),
                    "top_k_scores": torch.cat([x["top_k_scores"] for x in export_payload], dim=0),
                },
                export_topk_path,
            )
            logging.info(f"Saved recall top-{export_topk_k} candidates to {export_topk_path}")
