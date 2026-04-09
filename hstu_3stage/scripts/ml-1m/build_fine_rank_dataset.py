import argparse
import csv
import json
import os
import random
from typing import Dict, List, Tuple

import pandas as pd
import torch


def parse_sequence(raw_value) -> List[int]:
    if pd.isna(raw_value):
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    if text[0] in "[(" and text[-1] in "])":
        parsed = eval(text, {"__builtins__": {}})
        if isinstance(parsed, int):
            return [int(parsed)]
        return [int(x) for x in parsed]
    return [int(x) for x in text.split(",") if x]


def format_sequence(values: List[int]) -> str:
    return ",".join(str(int(x)) for x in values)


def is_positive_candidate(score_row: Dict[str, float], positive_threshold: int) -> bool:
    candidate_rating = score_row.get("candidate_rating")
    y_true = score_row.get("y_true")
    rating_positive = pd.notna(candidate_rating) and float(candidate_rating) > float(positive_threshold)
    y_true_positive = pd.notna(y_true) and float(y_true) > 0.0
    return bool(rating_positive or y_true_positive)


def build_score_frame(rank_payload: Dict[str, object]) -> Tuple[pd.DataFrame, str]:
    if {"user_id", "target_id", "target_rating", "y_true", "y_score"}.issubset(rank_payload.keys()):
        return (
            pd.DataFrame(
                {
                    "user_id": rank_payload["user_id"].tolist(),
                    "candidate_item": rank_payload["target_id"].tolist(),
                    "candidate_rating": rank_payload["target_rating"].tolist(),
                    "y_true": rank_payload["y_true"].tolist(),
                    "y_score": rank_payload["y_score"].tolist(),
                }
            ),
            "pointwise_scores",
        )

    if {"user_id", "target_id", "target_rating", "top_k_ids", "top_k_scores"}.issubset(rank_payload.keys()):
        rows: List[Dict[str, float]] = []
        user_ids = rank_payload["user_id"].tolist()
        target_ids = rank_payload["target_id"].tolist()
        target_ratings = rank_payload["target_rating"].tolist()
        top_k_ids = rank_payload["top_k_ids"].tolist()
        top_k_scores = rank_payload["top_k_scores"].tolist()
        for user_id, target_id, target_rating, candidate_ids, candidate_scores in zip(
            user_ids,
            target_ids,
            target_ratings,
            top_k_ids,
            top_k_scores,
        ):
            target_id = int(target_id)
            target_rating = int(target_rating)
            for candidate_item, candidate_score in zip(candidate_ids, candidate_scores):
                candidate_item = int(candidate_item)
                rows.append(
                    {
                        "user_id": int(user_id),
                        "candidate_item": candidate_item,
                        "candidate_rating": target_rating if candidate_item == target_id else 0,
                        "y_true": 1.0 if candidate_item == target_id else 0.0,
                        "y_score": float(candidate_score),
                    }
                )
        return pd.DataFrame(rows), "topk_scores"

    raise KeyError(
        "Unsupported rank_export_path payload. Expected either "
        "{user_id,target_id,target_rating,y_true,y_score} or "
        "{user_id,target_id,target_rating,top_k_ids,top_k_scores}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_csv", required=True)
    parser.add_argument("--rank_export_path", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--max_users", type=int, default=0)
    parser.add_argument("--top_m", type=int, default=1)
    parser.add_argument("--positive_threshold", type=int, default=3)
    parser.add_argument("--drop_users_without_positive", action="store_true")
    parser.add_argument(
        "--negative_sampling_mode",
        choices=["all", "random_one"],
        default="all",
        help="How to construct fine-stage negatives from the top_m candidate pool.",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--force_groundtruth_positive",
        action="store_true",
        help="When enabled, the selected groundtruth row in random_one mode is forced to be a positive label.",
    )
    args = parser.parse_args()
    if args.top_m <= 0:
        raise ValueError("--top_m must be a positive integer")

    base_frame = pd.read_csv(args.base_csv)
    rank_payload = torch.load(args.rank_export_path, map_location="cpu")

    score_frame, payload_format = build_score_frame(rank_payload)
    topm_frame = (
        score_frame.sort_values(["user_id", "y_score"], ascending=[True, False])
        .groupby("user_id", sort=False)
        .head(args.top_m)
    )
    topm_by_user: Dict[int, List[Dict[str, float]]] = {}
    for score_row in topm_frame.itertuples(index=False):
        user_id = int(score_row.user_id)
        topm_by_user.setdefault(user_id, []).append(
            {
                "candidate_item": score_row.candidate_item,
                "candidate_rating": score_row.candidate_rating,
                "y_true": score_row.y_true,
                "y_score": getattr(score_row, "y_score", 0.0),
            }
        )

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    stats = {
        "source_rows": int(len(base_frame)),
        "users_written": 0,
        "rows_written": 0,
        "users_missing_score": 0,
        "users_dropped_without_positive": 0,
        "positive_rows": 0,
        "negative_rows": 0,
        "top_m": int(args.top_m),
        "rank_payload_format": payload_format,
        "negative_sampling_mode": args.negative_sampling_mode,
        "force_groundtruth_positive": bool(args.force_groundtruth_positive),
    }
    rng = random.Random(args.random_seed)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "index",
                "user_id",
                "sequence_item_ids",
                "sequence_ratings",
                "sequence_timestamps",
            ],
        )
        writer.writeheader()

        for row in base_frame.itertuples(index=False):
            if args.max_users and stats["users_written"] >= args.max_users:
                break

            user_id = int(row.user_id)
            score_rows = topm_by_user.get(user_id)
            if not score_rows:
                stats["users_missing_score"] += 1
                continue

            seq_items = parse_sequence(row.sequence_item_ids)
            seq_ratings = parse_sequence(row.sequence_ratings)
            seq_timestamps = parse_sequence(row.sequence_timestamps)
            if len(seq_items) < 2 or len(seq_items) != len(seq_ratings) or len(seq_items) != len(seq_timestamps):
                continue

            history_items = seq_items[:-1]
            history_ratings = seq_ratings[:-1]
            history_timestamps = seq_timestamps[:-1]
            target_timestamp = int(seq_timestamps[-1])

            if args.drop_users_without_positive and not any(
                is_positive_candidate(score_row, args.positive_threshold) for score_row in score_rows
            ):
                stats["users_dropped_without_positive"] += 1
                continue

            selected_rows = score_rows
            if args.negative_sampling_mode == "random_one":
                positive_rows = [
                    score_row
                    for score_row in score_rows
                    if is_positive_candidate(score_row, args.positive_threshold)
                ]
                negative_rows = [
                    score_row
                    for score_row in score_rows
                    if not is_positive_candidate(score_row, args.positive_threshold)
                ]
                if not positive_rows:
                    stats["users_dropped_without_positive"] += 1
                    continue
                if not negative_rows:
                    stats["users_missing_score"] += 1
                    continue
                positive_rows = sorted(positive_rows, key=lambda row: float(row.get("y_score", 0.0)), reverse=True)
                selected_rows = [positive_rows[0], rng.choice(negative_rows)]

            for score_row in selected_rows:
                candidate_item = int(score_row["candidate_item"])
                candidate_rating = int(score_row["candidate_rating"])
                if args.force_groundtruth_positive and float(score_row.get("y_true", 0.0)) > 0.0:
                    candidate_rating = max(candidate_rating, int(args.positive_threshold) + 1)
                writer.writerow(
                    {
                        "index": stats["rows_written"],
                        "user_id": user_id,
                        "sequence_item_ids": format_sequence(history_items + [candidate_item]),
                        "sequence_ratings": format_sequence(history_ratings + [candidate_rating]),
                        "sequence_timestamps": format_sequence(history_timestamps + [target_timestamp]),
                    }
                )
                stats["rows_written"] += 1
                if candidate_rating > args.positive_threshold:
                    stats["positive_rows"] += 1
                else:
                    stats["negative_rows"] += 1

            stats["users_written"] += 1

    stats_path = f"{args.output_csv}.stats.json"
    with open(stats_path, "w", encoding="utf-8") as stats_file:
        json.dump(stats, stats_file, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"stats_path={stats_path}")


if __name__ == "__main__":
    main()
