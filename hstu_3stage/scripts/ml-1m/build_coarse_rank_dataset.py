import argparse
import csv
import json
import os
from typing import Dict, List

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


def unique_keep_order(values: List[int]) -> List[int]:
    seen = set()
    ordered = []
    for value in values:
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def load_topk_map(recall_topk_path: str) -> Dict[int, Dict[str, torch.Tensor]]:
    payload = torch.load(recall_topk_path, map_location="cpu")
    return {
        int(user_id): {
            "target_id": payload["target_id"][idx],
            "target_rating": payload["target_rating"][idx],
            "top_k_ids": payload["top_k_ids"][idx],
        }
        for idx, user_id in enumerate(payload["user_id"].tolist())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_csv", required=True)
    parser.add_argument("--recall_topk_path", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--candidate_k", type=int, default=1000)
    parser.add_argument("--max_users", type=int, default=0)
    parser.add_argument("--negative_rating", type=int, default=0)
    parser.add_argument("--positive_threshold", type=int, default=3)
    parser.add_argument("--append_target_if_missing", action="store_true")
    parser.add_argument("--drop_users_without_target", action="store_true")
    args = parser.parse_args()

    base_frame = pd.read_csv(args.base_csv)
    topk_by_user = load_topk_map(args.recall_topk_path)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    stats = {
        "source_rows": int(len(base_frame)),
        "users_written": 0,
        "rows_written": 0,
        "users_missing_export": 0,
        "users_target_in_candidates": 0,
        "users_target_missing": 0,
        "positive_rows": 0,
        "candidate_k": int(args.candidate_k),
    }

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

        for row_idx, row in enumerate(base_frame.itertuples(index=False)):
            if args.max_users and stats["users_written"] >= args.max_users:
                break

            user_id = int(row.user_id)
            export_row = topk_by_user.get(user_id)
            if export_row is None:
                stats["users_missing_export"] += 1
                continue

            seq_items = parse_sequence(row.sequence_item_ids)
            seq_ratings = parse_sequence(row.sequence_ratings)
            seq_timestamps = parse_sequence(row.sequence_timestamps)
            if len(seq_items) < 2 or len(seq_items) != len(seq_ratings) or len(seq_items) != len(seq_timestamps):
                continue

            history_items = seq_items[:-1]
            history_ratings = seq_ratings[:-1]
            history_timestamps = seq_timestamps[:-1]
            target_item = int(seq_items[-1])
            target_rating = int(seq_ratings[-1])
            target_timestamp = int(seq_timestamps[-1])

            candidate_ids = unique_keep_order(export_row["top_k_ids"][: args.candidate_k].tolist())
            target_in_candidates = target_item in candidate_ids
            if target_in_candidates:
                stats["users_target_in_candidates"] += 1
            else:
                stats["users_target_missing"] += 1
                if args.append_target_if_missing:
                    if len(candidate_ids) >= args.candidate_k and candidate_ids:
                        candidate_ids[-1] = target_item
                    else:
                        candidate_ids.append(target_item)
                    candidate_ids = unique_keep_order(candidate_ids)
                    target_in_candidates = target_item in candidate_ids
                if args.drop_users_without_target and not target_in_candidates:
                    continue

            for candidate_item in candidate_ids:
                candidate_rating = target_rating if candidate_item == target_item else args.negative_rating
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

            stats["users_written"] += 1

    stats_path = f"{args.output_csv}.stats.json"
    with open(stats_path, "w", encoding="utf-8") as stats_file:
        json.dump(stats, stats_file, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"stats_path={stats_path}")


if __name__ == "__main__":
    main()
