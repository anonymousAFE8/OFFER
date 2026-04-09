from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import pandas as pd
import torch

from generative_recommenders.research.data.my_dataset import MyDataset
from generative_recommenders.research.data.packed_rank_dataset import (
    PackedRankDataset,
    parse_sequence,
)


@dataclass
class FeatureDataset:
    feat_info_dict: dict
    feat_dim_dict: dict
    max_sequence_length: int
    num_unique_items: int
    max_item_id: int
    all_item_ids: List[int]
    num_unique_users: int
    max_user_id: int
    all_user_ids: List[int]
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset


def _collect_unique_values(sequences: Iterable[object]) -> Tuple[int, List[int]]:
    seen = set()
    for raw in sequences:
        seen.update(parse_sequence(raw))
    seen.discard(0)
    ordered = sorted(seen)
    return len(ordered), ordered


def _build_plain_sequence_dataset(
    log_file: str,
    eval_log_file: str,
    max_sequence_length: int,
    chronological: bool,
    exp_conf_dict: dict,
) -> FeatureDataset:
    frame = pd.read_csv(log_file)
    if "user_id" not in frame.columns or "sequence_item_ids" not in frame.columns:
        raise ValueError(f"Unexpected schema in {log_file}")

    num_unique_items, all_item_ids = _collect_unique_values(frame["sequence_item_ids"])
    all_user_ids = sorted(int(x) for x in frame["user_id"].dropna().unique().tolist())

    train_ignore_last_n = int(exp_conf_dict.get("train_ignore_last_n", 1))
    eval_ignore_last_n = int(exp_conf_dict.get("eval_ignore_last_n", 0))

    train_dataset = MyDataset(
        log_file=log_file,
        item_feat_file=None,
        user_feat_file=None,
        padding_length=max_sequence_length + 1,
        ignore_last_n=train_ignore_last_n,
        shift_id_by=0,
        chronological=chronological,
        exp_conf_dict=exp_conf_dict,
    )
    eval_dataset = MyDataset(
        log_file=eval_log_file,
        item_feat_file=None,
        user_feat_file=None,
        padding_length=max_sequence_length + 1,
        ignore_last_n=eval_ignore_last_n,
        shift_id_by=0,
        chronological=chronological,
        exp_conf_dict=exp_conf_dict,
    )

    return FeatureDataset(
        feat_info_dict={},
        feat_dim_dict={"item_feat": {}, "user_feat": {}},
        max_sequence_length=max_sequence_length,
        num_unique_items=num_unique_items,
        max_item_id=max(all_item_ids) if all_item_ids else 0,
        all_item_ids=all_item_ids,
        num_unique_users=len(all_user_ids),
        max_user_id=max(all_user_ids) if all_user_ids else 0,
        all_user_ids=all_user_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def _build_packed_rank_dataset(
    log_file: str,
    eval_log_file: str,
    max_sequence_length: int,
    chronological: bool,
    exp_conf_dict: dict,
) -> FeatureDataset:
    frame = pd.read_csv(log_file)
    if "candidate_item_ids" not in frame.columns:
        raise ValueError(f"Packed ranking dataset requires candidate_item_ids in {log_file}")

    item_sequences: List[object] = [frame["sequence_item_ids"]]
    item_sequences.append(frame["candidate_item_ids"])
    if "target_item_id" in frame.columns:
        item_sequences.append(frame["target_item_id"].astype(str))
    num_unique_items, all_item_ids = _collect_unique_values(
        pd.concat(item_sequences, ignore_index=True)
    )
    all_user_ids = sorted(int(x) for x in frame["user_id"].dropna().unique().tolist())

    train_ignore_last_n = int(exp_conf_dict.get("train_ignore_last_n", 0))
    eval_ignore_last_n = int(exp_conf_dict.get("eval_ignore_last_n", 0))

    train_dataset = PackedRankDataset(
        log_file=log_file,
        padding_length=max_sequence_length + 1,
        ignore_last_n=train_ignore_last_n,
        shift_id_by=0,
        chronological=chronological,
        exp_conf_dict=exp_conf_dict,
    )
    eval_dataset = PackedRankDataset(
        log_file=eval_log_file,
        padding_length=max_sequence_length + 1,
        ignore_last_n=eval_ignore_last_n,
        shift_id_by=0,
        chronological=chronological,
        exp_conf_dict=exp_conf_dict,
    )

    return FeatureDataset(
        feat_info_dict={},
        feat_dim_dict={"item_feat": {}, "user_feat": {}},
        max_sequence_length=max_sequence_length,
        num_unique_items=num_unique_items,
        max_item_id=max(all_item_ids) if all_item_ids else 0,
        all_item_ids=all_item_ids,
        num_unique_users=len(all_user_ids),
        max_user_id=max(all_user_ids) if all_user_ids else 0,
        all_user_ids=all_user_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def get_reco_dataset(
    dataset_name: str,
    max_sequence_length: int,
    chronological: bool,
    exp_conf_dict: dict = {},
):
    if dataset_name == "ml-1m":
        log_file = exp_conf_dict.get("log_file", "data/ml-1m/sasrec_format.csv")
        eval_log_file = exp_conf_dict.get("eval_log_file", log_file)
        if exp_conf_dict.get("packed_ranking", False):
            return _build_packed_rank_dataset(
                log_file=log_file,
                eval_log_file=eval_log_file,
                max_sequence_length=max_sequence_length,
                chronological=chronological,
                exp_conf_dict=exp_conf_dict,
            )
        return _build_plain_sequence_dataset(
            log_file=log_file,
            eval_log_file=eval_log_file,
            max_sequence_length=max_sequence_length,
            chronological=chronological,
            exp_conf_dict=exp_conf_dict,
        )

    if dataset_name == "KuaiRandPure":
        log_file = exp_conf_dict.get("log_file", "data/KuaiRandPure/sasrec_format.csv")
        train_dataset = MyDataset(
            log_file=log_file,
            item_feat_file="data/KuaiRandPure/map_item_feat.csv",
            user_feat_file="data/KuaiRandPure/map_user_feat.csv",
            padding_length=max_sequence_length + 1,
            ignore_last_n=1,
            shift_id_by=1,
            chronological=chronological,
            exp_conf_dict=exp_conf_dict,
        )
        eval_dataset = MyDataset(
            log_file=log_file,
            item_feat_file="data/KuaiRandPure/map_item_feat.csv",
            user_feat_file="data/KuaiRandPure/map_user_feat.csv",
            padding_length=max_sequence_length + 1,
            ignore_last_n=0,
            shift_id_by=1,
            chronological=chronological,
            exp_conf_dict=exp_conf_dict,
        )
        with open("data/KuaiRandPure/feat_stats.json", "r", encoding="utf-8") as f:
            feat_info_dict = json.load(f)
        return FeatureDataset(
            feat_info_dict=feat_info_dict,
            feat_dim_dict={
                "item_feat": {
                    "author_id": 32,
                    "video_type": 8,
                    "upload_dt": 8,
                    "upload_type": 8,
                    "music_id": 32,
                    "music_type": 8,
                    "tag": 8,
                },
                "user_feat": {
                    "user_active_degree": 8,
                    "is_live_streamer": 8,
                    "is_video_author": 8,
                    "follow_user_num_range": 8,
                    "fans_user_num_range": 8,
                    "friend_user_num_range": 8,
                    "register_days_range": 8,
                },
            },
            max_sequence_length=max_sequence_length,
            num_unique_items=7551,
            max_item_id=7583,
            all_item_ids=[x + 1 for x in range(7583)],
            num_unique_users=27077,
            max_user_id=27285,
            all_user_ids=[x + 1 for x in range(27285)],
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    if dataset_name == "QBArticle":
        log_file = exp_conf_dict.get("log_file", "data/QBArticle/sasrec_format.csv")
        train_dataset = MyDataset(
            log_file=log_file,
            item_feat_file="data/QBArticle/map_item_feat.csv",
            user_feat_file=None,
            padding_length=max_sequence_length + 1,
            ignore_last_n=1,
            shift_id_by=1,
            chronological=chronological,
            exp_conf_dict=exp_conf_dict,
        )
        eval_dataset = MyDataset(
            log_file=log_file,
            item_feat_file="data/QBArticle/map_item_feat.csv",
            user_feat_file=None,
            padding_length=max_sequence_length + 1,
            ignore_last_n=0,
            shift_id_by=1,
            chronological=chronological,
            exp_conf_dict=exp_conf_dict,
        )
        with open("data/QBArticle/feat_stats.json", "r", encoding="utf-8") as f:
            feat_info_dict = json.load(f)
        return FeatureDataset(
            feat_info_dict=feat_info_dict,
            feat_dim_dict={
                "item_feat": {
                    "exposure_count": 4,
                    "click_count": 4,
                    "like_count": 4,
                    "comment_count": 4,
                    "item_score1": 8,
                    "item_score2": 8,
                    "item_score3": 8,
                    "category_first": 8,
                    "category_second": 8,
                },
                "user_feat": {},
            },
            max_sequence_length=max_sequence_length,
            num_unique_items=7355,
            max_item_id=7355,
            all_item_ids=[x + 1 for x in range(7355)],
            num_unique_users=24516,
            max_user_id=24516,
            all_user_ids=[x + 1 for x in range(24516)],
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    if dataset_name == "AliEC":
        log_file = exp_conf_dict.get("log_file", "data/AliEC/sasrec_format.csv")
        train_dataset = MyDataset(
            log_file=log_file,
            item_feat_file="data/AliEC/map_item_feat.csv",
            user_feat_file=None,
            padding_length=max_sequence_length + 1,
            ignore_last_n=1,
            shift_id_by=1,
            chronological=chronological,
            exp_conf_dict=exp_conf_dict,
        )
        eval_dataset = MyDataset(
            log_file=log_file,
            item_feat_file="data/AliEC/map_item_feat.csv",
            user_feat_file=None,
            padding_length=max_sequence_length + 1,
            ignore_last_n=0,
            shift_id_by=1,
            chronological=chronological,
            exp_conf_dict=exp_conf_dict,
        )
        with open("data/AliEC/feat_stats.json", "r", encoding="utf-8") as f:
            feat_info_dict = json.load(f)
        return FeatureDataset(
            feat_info_dict=feat_info_dict,
            feat_dim_dict={
                "item_feat": {
                    "cate_id": 32,
                    "campaign_id": 16,
                    "customer": 16,
                    "brand": 16,
                    "price": 16,
                },
                "user_feat": {},
            },
            max_sequence_length=max_sequence_length,
            num_unique_items=108403,
            max_item_id=108403,
            all_item_ids=[x + 1 for x in range(108403)],
            num_unique_users=104657,
            max_user_id=104657,
            all_user_ids=[x + 1 for x in range(104657)],
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    if dataset_name == "Yelp":
        log_file = exp_conf_dict.get("log_file", "data/Yelp/sasrec_format.csv")
        train_dataset = MyDataset(
            log_file=log_file,
            item_feat_file="data/Yelp/map_item_feat.csv",
            user_feat_file=None,
            padding_length=max_sequence_length + 1,
            ignore_last_n=1,
            shift_id_by=1,
            chronological=chronological,
            exp_conf_dict=exp_conf_dict,
        )
        eval_dataset = MyDataset(
            log_file=log_file,
            item_feat_file="data/Yelp/map_item_feat.csv",
            user_feat_file=None,
            padding_length=max_sequence_length + 1,
            ignore_last_n=0,
            shift_id_by=1,
            chronological=chronological,
            exp_conf_dict=exp_conf_dict,
        )
        with open("data/Yelp/feat_stats.json", "r", encoding="utf-8") as f:
            feat_info_dict = json.load(f)
        return FeatureDataset(
            feat_info_dict=feat_info_dict,
            feat_dim_dict={
                "item_feat": {
                    "city": 16,
                    "state": 4,
                    "postal_code": 16,
                    "stars": 16,
                    "review_count": 16,
                    "is_open": 4,
                },
                "user_feat": {},
            },
            max_sequence_length=max_sequence_length,
            num_unique_items=26378,
            max_item_id=26378,
            all_item_ids=[x + 1 for x in range(26378)],
            num_unique_users=39665,
            max_user_id=39665,
            all_user_ids=[x + 1 for x in range(39665)],
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    raise ValueError(f"Unknown dataset {dataset_name}")
