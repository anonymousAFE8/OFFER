import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch


def parse_sequence(raw_value):
    text = str(raw_value).strip()
    if not text:
        return []
    return [int(x) for x in text.split(",") if x]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    hstu_root = repo_root / "hstu_3stage"
    base_csv = repo_root / "data" / "ml-1m" / "regen-050" / "sasrec_format_final_recall.csv"
    smoke_dir = hstu_root / "artifacts" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(base_csv)
    selected_rows = []
    seen_users = set()
    for row in frame.itertuples(index=False):
        user_id = int(row.user_id)
        if user_id in seen_users:
            continue
        seq_items = parse_sequence(row.sequence_item_ids)
        seq_ratings = parse_sequence(row.sequence_ratings)
        if len(seq_items) < 3:
            continue
        selected_rows.append((user_id, seq_items, seq_ratings))
        seen_users.add(user_id)
        if len(selected_rows) >= 4:
            break

    recall_payload = {
        "user_id": torch.tensor([row[0] for row in selected_rows], dtype=torch.int64),
        "target_id": torch.tensor([row[1][-1] for row in selected_rows], dtype=torch.int64),
        "target_rating": torch.tensor([row[2][-1] for row in selected_rows], dtype=torch.int64),
        "top_k_ids": torch.tensor(
            [
                [row[1][-1], row[1][-2], row[1][-3]]
                for row in selected_rows
            ],
            dtype=torch.int64,
        ),
    }
    recall_path = smoke_dir / "smoke_recall_topk.pt"
    torch.save(recall_payload, recall_path)

    coarse_csv = smoke_dir / "smoke_coarse.csv"
    subprocess.run(
        [
            sys.executable,
            str(hstu_root / "scripts" / "ml-1m" / "build_coarse_rank_dataset.py"),
            "--base_csv",
            str(base_csv),
            "--recall_topk_path",
            str(recall_path),
            "--output_csv",
            str(coarse_csv),
            "--candidate_k",
            "3",
            "--max_users",
            "4",
            "--drop_users_without_target",
        ],
        check=True,
        cwd=hstu_root,
    )

    coarse_frame = pd.read_csv(coarse_csv)
    grouped = []
    for user_id, group in coarse_frame.groupby("user_id", sort=False):
        target_row = group[group["sequence_ratings"].astype(str).str.endswith(",4") | group["sequence_ratings"].astype(str).str.endswith(",5")]
        if target_row.empty:
            target_row = group.iloc[[0]]
        target_item = int(parse_sequence(target_row.iloc[0]["sequence_item_ids"])[-1])
        target_rating = int(parse_sequence(target_row.iloc[0]["sequence_ratings"])[-1])
        candidate_items = [int(parse_sequence(v)[-1]) for v in group["sequence_item_ids"].tolist()]
        unique_items = []
        for item in candidate_items:
            if item not in unique_items:
                unique_items.append(item)
        top_k_ids = unique_items[:2]
        if target_item not in top_k_ids:
            top_k_ids = [target_item] + top_k_ids[:1]
        top_k_ids = top_k_ids[:2]
        top_k_scores = [1.0 if item == target_item else 0.1 for item in top_k_ids]
        grouped.append((int(user_id), target_item, target_rating, top_k_ids, top_k_scores))

    rank_payload = {
        "user_id": torch.tensor([row[0] for row in grouped], dtype=torch.int64),
        "target_id": torch.tensor([row[1] for row in grouped], dtype=torch.int64),
        "target_rating": torch.tensor([row[2] for row in grouped], dtype=torch.int64),
        "top_k_ids": torch.tensor([row[3] for row in grouped], dtype=torch.int64),
        "top_k_scores": torch.tensor([row[4] for row in grouped], dtype=torch.float32),
    }
    rank_path = smoke_dir / "smoke_coarse_scores.pt"
    torch.save(rank_payload, rank_path)

    fine_csv = smoke_dir / "smoke_fine.csv"
    subprocess.run(
        [
            sys.executable,
            str(hstu_root / "scripts" / "ml-1m" / "build_fine_rank_dataset.py"),
            "--base_csv",
            str(base_csv),
            "--rank_export_path",
            str(rank_path),
            "--output_csv",
            str(fine_csv),
            "--top_m",
            "2",
            "--drop_users_without_positive",
        ],
        check=True,
        cwd=hstu_root,
    )

    summary = {
        "base_csv": str(base_csv),
        "recall_payload": str(recall_path),
        "coarse_csv": str(coarse_csv),
        "fine_csv": str(fine_csv),
        "coarse_rows": int(sum(1 for _ in csv.DictReader(coarse_csv.open("r", encoding="utf-8")))),
        "fine_rows": int(sum(1 for _ in csv.DictReader(fine_csv.open("r", encoding="utf-8")))),
    }
    summary_path = smoke_dir / "verify_builder_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
