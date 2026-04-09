#专门用于处理特征的数据集，可以用于KuaiRandPure的处理
#需要注意的是，所有特征值都是从1开始的，0是padding部分。而所有item id和user id是从0开始的，在处理时要偏移一位。
#而特征文件中的item id和user id已经偏移过一位了，所以这其实是对齐的

import csv
import linecache

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


class MyDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        log_file: str,
        item_feat_file: str,
        user_feat_file: str,
        padding_length: int,
        ignore_last_n: int,  
        shift_id_by: int = 0,
        chronological: bool = False,
        exp_conf_dict: dict = {},   
    ) -> None:
        super().__init__()

        self.log_file = pd.read_csv(log_file,delimiter=",")
        if exp_conf_dict.get("use_feat"):
            if item_feat_file is not None:
               if "KuaiRandPure" in item_feat_file:
                  self.item_feat_table, self.item_feat_names = self.preprocess_id2feat(item_feat_file, index_name = "video_id")
               elif "AliEC" in item_feat_file:
                  self.item_feat_table, self.item_feat_names = self.preprocess_id2feat(item_feat_file, index_name = "adgroup_id")
               elif "Yelp" in item_feat_file:
                  self.item_feat_table, self.item_feat_names = self.preprocess_id2feat(item_feat_file, index_name = "business_id")
               else:
                  self.item_feat_table, self.item_feat_names = self.preprocess_id2feat(item_feat_file, index_name = "item_id")
            if user_feat_file is not None:
               self.user_feat_table, self.user_feat_names = self.preprocess_id2feat(user_feat_file, index_name = "user_id")
        self._padding_length: int = padding_length
        self._ignore_last_n: int = ignore_last_n
        self._cache: Dict[int, Dict[str, torch.Tensor]] = dict()
        self._shift_id_by: int = shift_id_by
        self._chronological: bool = chronological
        self.exp_conf_dict = exp_conf_dict
        self.use_user_feat = True if user_feat_file is not None else False
    
    # 读取特征csv文件并转成dataframe索引表加速特征索引
    def preprocess_id2feat(self, feat_file, index_name):
        df = pd.read_csv(feat_file)
        df = df.set_index(index_name)
        feat_names = df.columns
        if "tag" in feat_names:
            # 处理 tag 列：拆成 3 个列
            tag_split = (
                df["tag"].fillna("")
                .apply(lambda x: [int(v) for v in str(x).split(",") if v.strip().isdigit()])
            )
            tag_padded = tag_split.apply(lambda x: (x + [0,0,0])[:3])
            tag_df = pd.DataFrame(tag_padded.tolist(), index=df.index, columns=["tag1","tag2","tag3"])
            df = df.drop(columns=["tag"]).join(tag_df)
        padding_row = pd.DataFrame([[0] * df.shape[1]], columns=df.columns, index=[0])
        df = pd.concat([padding_row, df])
        return df, feat_names

    def __len__(self) -> int:
        return len(self.log_file)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self._cache.keys():
            return self._cache[idx]
        data = self.log_file.iloc[idx]
        sample = self.load_item(data)
        self._cache[idx] = sample
        return sample

    def load_item(self, data) -> Dict[str, torch.Tensor]:
        user_id = data.user_id

        # 这里对KuaiRand数据集要对user id也偏移一位
        if self._shift_id_by > 0:
            user_id += self._shift_id_by

        def eval_as_list(x: str, ignore_last_n: int) -> List[int]:
            y = eval(x)
            y_list = [y] if type(y) == int else list(y)
            if ignore_last_n > 0:
                # for training data creation
                y_list = y_list[:-ignore_last_n]
            return y_list

        def eval_int_list(
            x: str,
            target_len: int,
            ignore_last_n: int,
            shift_id_by: int,
        ) -> Tuple[List[int], int]:
            y = eval_as_list(x, ignore_last_n=ignore_last_n)
            y_len = len(y)
            y.reverse()
            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]
            return y, y_len

        if self.exp_conf_dict.get("only_pos", False) == True:
            sampling_kept_mask = [ x > self.exp_conf_dict["ratings_threshold"] for x in eval_as_list(data.sequence_ratings, self._ignore_last_n)]
            if sampling_kept_mask.count(True) <3:
                sampling_kept_mask = None

        all_seq, all_seq_len = eval_int_list(
            data.sequence_item_ids,
            self._padding_length,
            self._ignore_last_n,
            shift_id_by=self._shift_id_by,
        )
        all_actions, all_actions_len = eval_int_list(
            data.sequence_ratings,
            self._padding_length,
            self._ignore_last_n,
            shift_id_by=0,
        )

        # 处理数据集中没有时间戳的情况
        if hasattr(data, "sequence_timestamps"):
            all_timestamps, all_timestamps_len = eval_int_list(
                data.sequence_timestamps,
                self._padding_length,
                self._ignore_last_n,
                shift_id_by=0,
            )
        else:
            all_timestamps, all_timestamps_len = None, None

        def _truncate_or_pad_seq(
            y: List[int], target_len: int, chronological: bool
        ) -> List[int]:
            y_len = len(y)
            if y_len < target_len:
                y = y + [0] * (target_len - y_len)
            else:
                if not chronological:
                    y = y[:target_len]
                else:
                    y = y[-target_len:]
            assert len(y) == target_len
            return y

        seq_ids = all_seq[1:]
        actions = all_actions[1:]
        target_ids = all_seq[0]
        target_actions = all_actions[0]
        if all_timestamps is not None:
            timestamps = all_timestamps[1:]
            target_timestamps = all_timestamps[0]
        else:
            timestamps = None

        if self._chronological:
            seq_ids.reverse()
            actions.reverse()
            if timestamps is not None:
                timestamps.reverse()
            # 注意因为后面要用到all_seq得到特征，所以all_seq也需要做反转
            all_seq.reverse()

        max_seq_len = self._padding_length - 1
        seq_len = min(len(seq_ids), max_seq_len)
        seq_ids = _truncate_or_pad_seq(
            seq_ids,
            max_seq_len,
            self._chronological,
        )
        actions = _truncate_or_pad_seq(
            actions,
            max_seq_len,
            self._chronological,
        )
        if timestamps is not None:
            timestamps = _truncate_or_pad_seq(
                timestamps,
                max_seq_len,
                self._chronological,
            )
        # 注意因为后面要用到all_seq得到特征，所以all_seq也需要做补全
        all_seq = _truncate_or_pad_seq(
            all_seq,
            self._padding_length,
            self._chronological,
        )

        #该函数用于id映射特征
        def look_for_feat(feat_names, seq_ids, id2feat_dict, return_list):
            feat_dict = {}
            for name in feat_names:
                feat_list = []
                for idx in seq_ids:
                    if idx == 0:
                        if name == "tag":
                           feat_list.append([0,0,0])
                        else:
                           feat_list.append(0)
                    else:
                        if name == "tag":
                            val = id2feat_dict[idx][name]
                            if pd.isna(val) or str(val).strip() == "":
                               val_list = []
                            else:
                               val_list = [int(x) for x in str(val).split(",") if x.strip().isdigit()]
                            if len(val_list) < 3:
                               val_list += [0] * (3 - len(val_list))
                            feat_list.append(val_list)
                        else:
                            val = id2feat_dict[idx][name]
                            if pd.isna(val):
                               val = 0
                            feat_list.append(val)
                feat_dict[name] = feat_list if return_list else feat_list[0]
            return feat_dict
        
        def look_for_feat_fast(seq_ids, df, return_list=True):
            sub = df.loc[seq_ids].fillna(0)
            if return_list:
                return sub.values.tolist()
            else:
                return sub.iloc[0].to_dict()
        
        if self.exp_conf_dict.get("use_feat"):
            item_feat = self.look_for_feat_fast_dict(all_seq, feat_type = "item")
            if self.use_user_feat:
                user_feat = self.look_for_feat_fast_dict(user_id, feat_type = "user")
            item_feat = { name: torch.tensor(feat, dtype=torch.int64) for name, feat in item_feat.items()}

        ret = {
            "user_id": user_id,
            "seq_ids": torch.tensor(seq_ids, dtype=torch.int64),
            "actions": torch.tensor(actions, dtype=torch.int64),
            "seq_len": seq_len,
            "target_ids": target_ids,
            "target_actions": target_actions,
        }
        if timestamps is not None:
            ret["timestamps"] = torch.tensor(timestamps, dtype=torch.int64)
            ret["target_timestamps"] = target_timestamps
        if self.exp_conf_dict.get("use_feat"):
            ret["item_feat"] = item_feat
            if self.use_user_feat:
               ret["user_feat"] = user_feat
        return ret

    # 返回字典
    def look_for_feat_fast_dict(self, seq_ids, feat_type, re_feat_name = False):
        if feat_type == "item":
            feat_names = self.item_feat_names
            df = self.item_feat_table
        elif feat_type == "user":
            feat_names = self.user_feat_names
            df = self.user_feat_table
        sub = df.loc[seq_ids].fillna(0)
        feat_dict = {}
        for name in feat_names:
            if name == "tag":
                tag_values = df.loc[seq_ids, ["tag1", "tag2", "tag3"]].values.tolist()
                feat_dict[name] = tag_values
            else:
                feat_dict[name] = sub[name].tolist()
        return (feat_dict, feat_names) if re_feat_name else feat_dict
