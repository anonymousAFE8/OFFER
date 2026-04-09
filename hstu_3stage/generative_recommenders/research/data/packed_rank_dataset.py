from __future__ import annotations

import ast
from typing import Dict, List, Optional

import pandas as pd
import torch


def parse_sequence(raw_value: object) -> List[int]:
    if pd.isna(raw_value):
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    value = ast.literal_eval(text)
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    raise ValueError(f"Unsupported sequence payload: {raw_value!r}")


def pad_sequence(values: List[int], target_len: int) -> List[int]:
    if len(values) >= target_len:
        return values[:target_len]
    return values + [0] * (target_len - len(values))


class PackedRankDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        log_file: str,
        padding_length: int,
        ignore_last_n: int,
        shift_id_by: int = 0,
        chronological: bool = False,
        exp_conf_dict: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.log_file = pd.read_csv(log_file, delimiter=",")
        self._padding_length = padding_length
        self._ignore_last_n = ignore_last_n
        self._shift_id_by = shift_id_by
        self._chronological = chronological
        self.exp_conf_dict = exp_conf_dict or {}
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._packed_candidate_k = int(self.exp_conf_dict.get("packed_candidate_k", 0))

    def __len__(self) -> int:
        return len(self.log_file)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self._cache:
            return self._cache[idx]
        row = self.log_file.iloc[idx]
        sample = self.load_item(row)
        self._cache[idx] = sample
        return sample

    def load_item(self, data) -> Dict[str, torch.Tensor]:
        user_id = int(data.user_id)
        if self._shift_id_by > 0:
            user_id += self._shift_id_by

        history_ids = parse_sequence(data.sequence_item_ids)
        history_actions = parse_sequence(data.sequence_ratings)
        history_timestamps = parse_sequence(data.sequence_timestamps)
        if self._ignore_last_n > 0:
            history_ids = history_ids[:-self._ignore_last_n]
            history_actions = history_actions[:-self._ignore_last_n]
            history_timestamps = history_timestamps[:-self._ignore_last_n]

        if not (
            len(history_ids) == len(history_actions) == len(history_timestamps)
        ):
            raise ValueError("History sequence fields must have the same length")

        if self._chronological:
            ordered_ids = history_ids
            ordered_actions = history_actions
            ordered_timestamps = history_timestamps
        else:
            ordered_ids = list(reversed(history_ids))
            ordered_actions = list(reversed(history_actions))
            ordered_timestamps = list(reversed(history_timestamps))

        max_seq_len = self._padding_length - 1
        seq_len = min(len(ordered_ids), max_seq_len)
        ordered_ids = pad_sequence(ordered_ids, max_seq_len)
        ordered_actions = pad_sequence(ordered_actions, max_seq_len)
        ordered_timestamps = pad_sequence(ordered_timestamps, max_seq_len)

        candidate_ids = parse_sequence(data.candidate_item_ids)
        candidate_actions = parse_sequence(data.candidate_ratings)
        if len(candidate_ids) != len(candidate_actions):
            raise ValueError("Candidate ids and ratings must have the same length")
        if self._shift_id_by > 0:
            candidate_ids = [x + self._shift_id_by if x > 0 else 0 for x in candidate_ids]
        candidate_len = len(candidate_ids)
        packed_candidate_k = max(self._packed_candidate_k, candidate_len)
        candidate_mask = [1] * candidate_len + [0] * max(0, packed_candidate_k - candidate_len)
        candidate_ids = pad_sequence(candidate_ids, packed_candidate_k)
        candidate_actions = pad_sequence(candidate_actions, packed_candidate_k)

        target_item_id = int(data.target_item_id)
        if self._shift_id_by > 0 and target_item_id > 0:
            target_item_id += self._shift_id_by
        target_rating = int(data.target_rating)
        target_timestamp = int(data.target_timestamp)

        return {
            "user_id": user_id,
            "seq_ids": torch.tensor(ordered_ids, dtype=torch.int64),
            "actions": torch.tensor(ordered_actions, dtype=torch.int64),
            "seq_len": seq_len,
            "timestamps": torch.tensor(ordered_timestamps, dtype=torch.int64),
            "target_ids": target_item_id,
            "target_actions": target_rating,
            "target_timestamps": target_timestamp,
            "candidate_ids": torch.tensor(candidate_ids, dtype=torch.int64),
            "candidate_actions": torch.tensor(candidate_actions, dtype=torch.int64),
            "candidate_mask": torch.tensor(candidate_mask, dtype=torch.bool),
            "candidate_len": candidate_len,
        }
