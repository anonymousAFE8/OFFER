# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import abc
import logging
import torch
from collections import OrderedDict

from generative_recommenders.research.modeling.initialization import truncated_normal


class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor, item_ratios=None) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        num_users: int,
        item_embedding_dim: int,
        feat_embedding_dim: int,
        exp_conf_dict: dict,
        feat_info_dict: dict, #会传入一个特征字典，里面有用户和物品的特征信息，包含了特征范围和数量
        feat_dim_dict: dict, #预先定义好的特征维度
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self.exp_conf_dict: dict = exp_conf_dict
        use_feat = bool(exp_conf_dict.get("use_feat"))
        item_feat_dim_dict = (feat_dim_dict or {}).get("item_feat", {})
        item_feat_info_dict = (feat_info_dict or {}).get("item_feat", {})
        if exp_conf_dict["load_embs"] == False or "cat" in exp_conf_dict["load_type"]:
            self._item_emb = torch.nn.Embedding( num_items + 1, item_embedding_dim, padding_idx=0)
            if use_feat:
                self._feat_emb = torch.nn.ModuleDict()
                for feat_name, feat_info in item_feat_info_dict.items():
                    if feat_name == "video_id":
                        continue
                    if feat_name == "item_id":
                        continue
                    if feat_name == "adgroup_id":
                        continue
                    if feat_name == "business_id":
                        continue
                    feat_emb = torch.nn.Embedding( int(feat_info["max"]) + 1, item_feat_dim_dict[feat_name], padding_idx=0)  
                    self._feat_emb[feat_name] = feat_emb
                self.down_mlp = torch.nn.Linear(item_embedding_dim + sum(item_feat_dim_dict.values()), item_embedding_dim)
        if exp_conf_dict["load_embs"] == True:
            self._item_pretrain_emb = torch.nn.Embedding(num_items + 1, item_embedding_dim, padding_idx=0)
            if use_feat:
                self._feat_pretrain_emb = torch.nn.ModuleDict()
                self.down_mlp = torch.nn.Linear(item_embedding_dim + sum(item_feat_dim_dict.values()), item_embedding_dim)
                for feat_name, feat_info in item_feat_info_dict.items():
                    if feat_name == "video_id":
                        continue
                    if feat_name == "item_id":
                        continue
                    if feat_name == "adgroup_id":
                        continue
                    if feat_name == "business_id":
                        continue
                    feat_emb = torch.nn.Embedding( int(feat_info["max"]) + 1, item_feat_dim_dict[feat_name], padding_idx=0)
                    self._feat_pretrain_emb[feat_name] = feat_emb
            pretrain_dict = torch.load(exp_conf_dict["model_dir"])
            embedding_weights = OrderedDict() 
            for key, value in pretrain_dict.items():
                if "item_emb" in key:
                    embedding_weights["weight"] = value
            self._item_pretrain_emb.load_state_dict(embedding_weights)
            
            if use_feat:
                for key, value in pretrain_dict.items():
                    embedding_weights = OrderedDict()
                    if "feat_emb" in key:
                        for name in item_feat_info_dict.keys():
                            if name in key:
                                feat_name = name
                                break
                        embedding_weights["weight"] = value
                        self._feat_pretrain_emb[feat_name].load_state_dict(embedding_weights)

                down_mlp_weights = OrderedDict()
                for key, value in pretrain_dict.items():
                    if "down_mlp.weight" in key:
                        down_mlp_weights["weight"] = value
                    elif "down_mlp.bias" in key:
                        down_mlp_weights["bias"] = value
                if down_mlp_weights:
                    self.down_mlp.load_state_dict(down_mlp_weights)

            #是否停止embedding表的梯度更新
            if "close" in exp_conf_dict["load_type"]:
                for param in self._item_recall_emb.parameters():
                    param.requires_grad = False
                for param in self._feat_recall_emb.parameters():
                    param.requires_grad = False
            if "cat" in exp_conf_dict["load_type"]:
                self._emb_cat_lin = torch.nn.Linear(item_embedding_dim * 2, item_embedding_dim)

        if self.exp_conf_dict.get("use_user_id", False) == True:
            load_user_id_type = exp_conf_dict.get("load_user_id_type") or ""
            if exp_conf_dict.get("load_embs", False) == False or "new" in load_user_id_type:
                self._user_emb = torch.nn.Embedding( num_users + 1, item_embedding_dim, padding_idx=0)
                logging.info("use user id")
            else:
                self._user_emb = torch.nn.Embedding( num_users + 1, item_embedding_dim, padding_idx=0)
                pretrained_dict = torch.load(exp_conf_dict["model_dir"])
                embedding_weights = OrderedDict() 
                for key, value in pretrained_dict.items():
                    if "user_emb" in key:
                        embedding_weights["weight"] = value
                self._user_emb.load_state_dict(embedding_weights)
                if "close" in load_user_id_type:
                    for param in self._user_emb.parameters():
                        param.requires_grad = False
                    logging.info("close recall param grad")
                logging.info("use the recall user id")
        #self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    # def reset_params(self) -> None:
    #     for name, params in self.named_parameters():
    #         if any(x in name for x in ["_item_emb"]):
    #             truncated_normal(params, mean=0.0, std=0.02)
    #             logging.info(f"Initialize {name} as truncated normal: {params.data.size()} params")
    #         else:
    #             logging.info(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor, item_actions, user_id, feat: dict) -> torch.Tensor:
        if self.exp_conf_dict["load_embs"] == False or "cat" in self.exp_conf_dict["load_type"]:
            rank_emb = self._item_emb(item_ids)

            if self.exp_conf_dict.get("use_feat"):
                feat_emb_list = [rank_emb]
                for feat_name, feat_val in feat.items():
                    feat_emb = self._feat_emb[feat_name](feat_val)
                    if len(feat_emb.shape) == 4:
                        feat_emb = feat_emb.mean(dim=2)
                    feat_emb_list.append(feat_emb)
                rank_emb = torch.cat(feat_emb_list, dim = -1)
                rank_emb = self.down_mlp(rank_emb)

        if self.exp_conf_dict["load_embs"] == True:
            recall_emb = self._item_pretrain_emb(item_ids)

            if self.exp_conf_dict.get("use_feat"):
                feat_emb_list = [recall_emb]
                for feat_name, feat_val in feat.items():
                    feat_emb = self._feat_pretrain_emb[feat_name](feat_val)
                    if len(feat_emb.shape) == 4:
                        feat_emb = feat_emb.mean(dim=2)
                    feat_emb_list.append(feat_emb)
                recall_emb = torch.cat(feat_emb_list, dim = -1)
                recall_emb = self.down_mlp(recall_emb)

            if "cat" in self.exp_conf_dict["load_type"]:
                item_emb = self._emb_cat_lin(torch.cat([rank_emb, recall_emb],dim=-1))
            elif "inh" in self.exp_conf_dict["load_type"]:
                item_emb = recall_emb
            else:
                item_emb = recall_emb
        else:
            item_emb = rank_emb 

        if self.exp_conf_dict.get("use_user_id", False) == True and user_id is not None:
            item_emb = torch.cat([self._user_emb(user_id).unsqueeze(1), item_emb], dim=1)
        return item_emb

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
    
class InterEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        exp_conf_dict: dict,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self._ratio_emb = torch.nn.Embedding(
            exp_conf_dict["num_actions"], item_embedding_dim, padding_idx=0
        )
        self.reset_params()
        self.exp_conf_dict = exp_conf_dict
        
    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if any(x in name for x in ["_item_emb", "_ratio_emb"]):
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor, item_actions: torch.Tensor) -> torch.Tensor:
        item_emb = self._item_emb(item_ids)
        actions_emb = self._ratio_emb(item_actions)
        B, N, D = item_emb.shape
        if self.exp_conf_dict["modify_action"] == False:
            ret_emb = torch.stack([item_emb, actions_emb], dim=2).view(B, 2*N, D)
        else:
            ret_emb = torch.stack([item_emb, actions_emb + item_emb], dim=2).view(B, 2*N, D)
        return ret_emb

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim

# class AddActionEmbeddingModule(EmbeddingModule):
#     def __init__(
#         self,
#         num_items: int,
#         item_embedding_dim: int,
#         exp_conf_dict: dict,
#     ) -> None:
#         super().__init__()

#         self._item_embedding_dim: int = item_embedding_dim
#         self._item_emb = torch.nn.Embedding(
#             num_items + 1, item_embedding_dim, padding_idx=0
#         )
#         self._action_emb = torch.nn.Embedding(
#             exp_conf_dict["num_actions"] + 1, item_embedding_dim, padding_idx=0
#         )
#         #self.proj = torch.nn.Linear(self._item_embedding_dim * 2, self._item_embedding_dim)
#         if exp_conf_dict.get("use_user_id", False) == True:
#             self._user_emb = torch.nn.Embedding(
#                 6040 + 1, item_embedding_dim, padding_idx=0
#             )
#             logging.info("Use the user id in front of embedding")
#         self.exp_conf_dict = exp_conf_dict
#         logging.info("Use the add-action embedding")
#         self.reset_params()
        
#     def debug_str(self) -> str:
#         return f"local_emb_d{self._item_embedding_dim}"

#     def reset_params(self) -> None:
#         for name, params in self.named_parameters():
#             if any(x in name for x in ["_item_emb", "_action_emb", "_user_emb", "proj"]):
#                 truncated_normal(params, mean=0.0, std=0.02)
#                 logging.info(f"Initialize {name} as truncated normal: {params.shape} params")
#             else:
#                 logging.info(f"Skipping initializing params {name} - not configured")

#     def get_item_embeddings(self, item_ids: torch.Tensor, item_actions: torch.Tensor, user_id: torch.Tensor) -> torch.Tensor:
#         item_emb = self._item_emb(item_ids)
#         if item_actions is not None:
#             actions_emb = self._action_emb(item_actions)
#             #print(actions_emb)
#             #gate = torch.sigmoid(self.proj(torch.cat([item_emb, actions_emb], dim=-1)))
#             #item_emb = (1 - gate) * item_emb + gate * actions_emb
#             item_emb = item_emb + actions_emb
#         if self.exp_conf_dict.get("use_user_id", False) == True and user_id is not None:
#             item_emb = torch.cat([self._user_emb(user_id).unsqueeze(1), item_emb], dim=1)
#         return item_emb

#     @property
#     def item_embedding_dim(self) -> int:
#         return self._item_embedding_dim
