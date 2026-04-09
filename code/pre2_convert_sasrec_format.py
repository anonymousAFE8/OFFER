import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


pattern_dict = {}
pat_key_dict = {}


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TrieNode:
    def __init__(self):
        self.children = {}
        self.pattern_info = None


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, pattern, pattern_info):
        node = self.root
        for item in pattern:
            if item not in node.children:
                node.children[item] = TrieNode()
            node = node.children[item]
        node.pattern_info = pattern_info

    def search(self, sequence):
        matches = []
        for i in range(len(sequence)):
            node = self.root
            j = i
            while j < len(sequence) and sequence[j] in node.children:
                node = node.children[sequence[j]]
                if node.pattern_info is not None:
                    matches.append((i, node.pattern_info))
                j += 1
        return matches


def create_filter_stats():
    return {
        "total_sequences": 0,
        "changed_sequences": 0,
        "unchanged_sequences": 0,
        "total_original_items": 0,
        "total_processed_items": 0,
        "total_matches": 0,
        "total_replaced_items": 0,
        "changed_sequence_lengths": [],
        "changed_match_counts": [],
        "changed_replaced_item_counts": [],
    }


def update_filter_stats(stats, original_length, processed_length, match_stats):
    stats["total_sequences"] += 1
    stats["total_original_items"] += original_length
    stats["total_processed_items"] += processed_length
    stats["total_matches"] += match_stats["num_matches"]
    stats["total_replaced_items"] += match_stats["num_replaced_items"]

    if match_stats["changed"]:
        stats["changed_sequences"] += 1
        stats["changed_sequence_lengths"].append(original_length)
        stats["changed_match_counts"].append(match_stats["num_matches"])
        stats["changed_replaced_item_counts"].append(match_stats["num_replaced_items"])
    else:
        stats["unchanged_sequences"] += 1


def finalize_filter_stats(stats):
    changed_sequences = stats["changed_sequences"]
    total_sequences = stats["total_sequences"]
    total_original_items = stats["total_original_items"]

    def safe_mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "total_sequences": total_sequences,
        "changed_sequences": changed_sequences,
        "unchanged_sequences": stats["unchanged_sequences"],
        "filtered_sequence_percentage": changed_sequences / total_sequences if total_sequences else 0.0,
        "total_original_items": total_original_items,
        "total_processed_items": stats["total_processed_items"],
        "compression_ratio_items": (
            stats["total_processed_items"] / total_original_items if total_original_items else 0.0
        ),
        "total_matches": stats["total_matches"],
        "total_replaced_items": stats["total_replaced_items"],
        "avg_matches_per_changed_sequence": (
            stats["total_matches"] / changed_sequences if changed_sequences else 0.0
        ),
        "avg_replaced_items_per_changed_sequence": (
            stats["total_replaced_items"] / changed_sequences if changed_sequences else 0.0
        ),
        "avg_original_length_changed_sequences": safe_mean(stats["changed_sequence_lengths"]),
        "avg_matches_changed_sequences": safe_mean(stats["changed_match_counts"]),
        "avg_replaced_items_changed_sequences": safe_mean(stats["changed_replaced_item_counts"]),
    }


def match_and_replace_pattern(sequence_items, sequence_ratings, sequence_timestamps, trie, new_id_start):
    processed_sequence = list(sequence_items)
    matches = trie.search(processed_sequence)
    matches.sort(key=lambda x: (x[0], -len(x[1]["pattern"])))

    non_overlapping_matches = []
    last_pos = -1

    for pos, pattern_info in matches:
        pat_len = len(pattern_info["pattern"])
        if pos >= last_pos:
            non_overlapping_matches.append((pos, pattern_info))
            last_pos = pos + pat_len

    matched_spans = [(pos, pos + len(pattern_info["pattern"])) for pos, pattern_info in non_overlapping_matches]
    num_matches = len(non_overlapping_matches)
    num_replaced_items = sum(len(pattern_info["pattern"]) for _, pattern_info in non_overlapping_matches)

    non_overlapping_matches.reverse()
    for pos, pattern_info in non_overlapping_matches:
        pat_key = (tuple(pattern_info["pattern"]), tuple(pattern_info["sequence_ratings"]))
        pat_len = len(pattern_info["pattern"])

        if pat_key not in pat_key_dict:
            if not pattern_dict:
                pat_key_dict[pat_key] = new_id_start
                pattern_dict[new_id_start] = {
                    "pattern": processed_sequence[pos : (pos + pat_len)],
                    "sequence_ratings": sequence_ratings[pos : (pos + pat_len)],
                    "sequence_timestamps": sequence_timestamps[pos : (pos + pat_len)],
                }
            else:
                max_pd_id = max(pattern_dict.keys()) + 1
                pat_key_dict[pat_key] = max_pd_id
                pattern_dict[max_pd_id] = {
                    "pattern": processed_sequence[pos : (pos + pat_len)],
                    "sequence_ratings": sequence_ratings[pos : (pos + pat_len)],
                    "sequence_timestamps": sequence_timestamps[pos : (pos + pat_len)],
                }

        processed_sequence = processed_sequence[:pos] + [pat_key_dict[pat_key]] + processed_sequence[pos + pat_len :]
        sequence_ratings = (
            sequence_ratings[:pos]
            + [int(np.mean(sequence_ratings[pos : (pos + pat_len)]))]
            + sequence_ratings[pos + pat_len :]
        )
        sequence_timestamps = (
            sequence_timestamps[:pos]
            + [int(np.mean(sequence_timestamps[pos : (pos + pat_len)]))]
            + sequence_timestamps[pos + pat_len :]
        )

    match_stats = {
        "changed": num_matches > 0,
        "num_matches": num_matches,
        "num_replaced_items": num_replaced_items,
        "matched_spans": matched_spans,
    }
    return processed_sequence, sequence_ratings, sequence_timestamps, match_stats


def main():
    parser = argparse.ArgumentParser(description="Process data with root path.")
    parser.add_argument("--root_path", type=str, required=True, help="The root directory path for reading and writing files.")
    args = parser.parse_args()

    tokens_file_path = os.path.join(args.root_path, "restored_tokens.json")
    csv_input_path = os.path.join(args.root_path, "sasrec_format.csv")
    csv_output_path = os.path.join(args.root_path, "sasrec_format.csv")
    pattern_dict_path = os.path.join(args.root_path, "pattern_mappings.json")
    filter_stats_path = os.path.join(args.root_path, "pattern_filter_stats.json")

    patterns = load_json(tokens_file_path)

    trie = Trie()
    for pattern in patterns:
        pattern_int = [int(each_token) for each_token in pattern]
        pattern_info = {
            "pattern": pattern_int,
            "sequence_ratings": [0] * len(pattern),
            "sequence_timestamps": [0] * len(pattern),
        }
        trie.insert(pattern_int, pattern_info)

    df = pd.read_csv(csv_input_path)

    new_data = []
    new_id_start = 0
    filter_stats = create_filter_stats()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scanning max item id"):
        sequence_items = list(map(int, row["sequence_item_ids"].split(",")))
        if max(sequence_items) > new_id_start:
            new_id_start = max(sequence_items)
    new_id_start += 1

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sequences"):
        sequence_items = list(map(int, row["sequence_item_ids"].split(",")))
        ratings = list(map(int, row["sequence_ratings"].split(",")))
        timestamps = list(map(int, row["sequence_timestamps"].split(",")))

        processed_sequence, processed_ratings, processed_timestamps, match_stats = match_and_replace_pattern(
            sequence_items, ratings, timestamps, trie, new_id_start
        )

        update_filter_stats(
            filter_stats,
            original_length=len(sequence_items),
            processed_length=len(processed_sequence),
            match_stats=match_stats,
        )

        new_row = {
            "index": row["index"],
            "user_id": row["user_id"],
            "sequence_item_ids": ",".join(map(str, processed_sequence)),
            "sequence_ratings": ",".join(map(str, processed_ratings)),
            "sequence_timestamps": ",".join(map(str, processed_timestamps)),
        }
        new_data.append(new_row)

    print(new_data[0])
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(csv_output_path, index=False)

    with open(pattern_dict_path, "w", encoding="utf-8") as f:
        json.dump(pattern_dict, f, indent=4)

    with open(filter_stats_path, "w", encoding="utf-8") as f:
        json.dump(finalize_filter_stats(filter_stats), f, indent=4)


if __name__ == "__main__":
    main()
