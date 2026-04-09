import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import json

def key_to_tuple(k: str) -> tuple:
    return tuple(k.split("|"))

def load_info_metrics_json(file_path: str):
    try:
        with open(file_path, 'r') as f:
            loaded_metrics_str_keys = json.load(f)
        mi_str_keys = loaded_metrics_str_keys.get('mutual_information', {})
        mi_tuple_keys = {key_to_tuple(k): v for k, v in mi_str_keys.items()}
        info_metrics = {
            'entropy': loaded_metrics_str_keys.get('entropy', {}),
            'mutual_information': mi_tuple_keys
        }
        print(f"Successfully loaded and parsed info metrics from: {file_path}")
        return info_metrics
    except FileNotFoundError:
        print(f"Error: Precomputed metrics file not found at {file_path}")
        raise

# 特征生成部分 按照指定概率 指定特征名 进行生成式学习
class GenFeatureModule(nn.Module):
    def __init__(
        self,
        item_emb: nn.Embedding,
        feat_emb: nn.ModuleDict,
        feat_names: List[str],
        hidden_dim: int,
        proj_dim: int,
        feat_pre_info_dict: dict,
        loss_type: str = "contrastive",
        temperature: float = 0.1,
        mask_mode: str = "hard",       # mask类型：hard or soft
        dropout_p: float = 0.2,         # soft mask dropout
        sampling_temp: float = 1.0,
        num_masked_features: int = 1,
        lambda_val: float = 0.2,
    ):
        super().__init__()
        self.item_emb = item_emb
        self.feat_emb = feat_emb
        self.feat_names = feat_names
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.loss_type = loss_type
        self.temperature = temperature
        self.feat_pre_info_dict = feat_pre_info_dict
        self.mask_mode = mask_mode
        self.num_masked_features = num_masked_features
        self.lambda_val = lambda_val
        if mask_mode == "soft":
           self.dropout = nn.Dropout(dropout_p)

        input_dim = item_emb.embedding_dim + sum([emb.embedding_dim for emb in feat_emb.values()])
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

        self.pred_heads = nn.ModuleDict({
            name: nn.Linear(proj_dim, emb.embedding_dim)
            for name, emb in feat_emb.items() if name in feat_names
        })

        self.info_metrics = load_info_metrics_json(self.feat_pre_info_dict)

        self.entropy = self.info_metrics['entropy']
        self.mutual_info = self.info_metrics['mutual_information']

        self.sampling_temp = sampling_temp
    
    # prob sample
    def sample_feat_mask(self, device):
        mask = {}
        for name in self.feat_names:
            p = self.feat_prob.get(name, 0.0)  
            mask[name] = torch.rand(1, device=device).item() < p
        return mask
    
    # info driven sample feat
    def sample_feat_mask_info_driven(self, device) -> Dict[str, bool]:
        masked_subset = []
        
        candidate_features = self.feat_names.copy()
        for _ in range(self.num_masked_features):
            if not candidate_features:
                break
                
            marginal_gains = []
            features_with_gain = []
            
            # Calculate the "context-aware" gain for each remaining candidate
            for f_k in candidate_features:
                # H(F_k)
                entropy_k = self.entropy.get(f_k, 0.0)
                
                # **CRITICAL CHANGE**: Redundancy is now calculated against ALL OTHER *REMAINING* CANDIDATES
                # which are the current context for this selection step.
                redundancy = 0.0
                # The context for f_k are all other features still in the candidate list
                context_features = [f for f in candidate_features if f != f_k]
                
                for f_j in context_features:
                    mi_val = self.mutual_info.get((f_k, f_j), self.mutual_info.get((f_j, f_k), 0.0))
                    redundancy += mi_val
                
                # The value of selecting f_k is its entropy minus its redundancy with the context
                gain = entropy_k - self.lambda_val * redundancy
                marginal_gains.append(gain)
                features_with_gain.append(f_k)
            
            # Convert gains to probabilities using Softmax
            gains_tensor = torch.tensor(marginal_gains, dtype=torch.float)
            probs = F.softmax(gains_tensor / self.sampling_temp, dim=-1)
            
            # Sample the next feature to be masked
            sampled_index = torch.multinomial(probs, 1).item()
            next_feature_to_mask = features_with_gain[sampled_index]
            
            # Update sets
            masked_subset.append(next_feature_to_mask)
            candidate_features.remove(next_feature_to_mask) # This feature is now masked, not a candidate anymore
            
        # Convert the list of masked features to the final boolean mask dictionary
        final_mask = {name: (name in masked_subset) for name in self.feat_names}
        return final_mask

    def forward(
        self,
        item_id: torch.Tensor,
        feat: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        B, N = item_id.shape
        device = item_id.device

        #feat_mask = self.sample_feat_mask(device)

        # use info-drive method:
        feat_mask = self.sample_feat_mask_info_driven(device)
            
        item_e = self.item_emb(item_id)

        # 拼接除目标特征外的embedding（可以选择是否排除目标特征）
        feat_inputs = []
        for name, ids in feat.items():
            feat_emb = self.feat_emb[name](ids)
            if name == "tag":
                feat_emb = feat_emb.mean(dim=2)

            if name in feat_mask.keys() and feat_mask[name]:
                if self.mask_mode == "hard":
                    feat_emb = torch.zeros_like(feat_emb)

                elif self.mask_mode == "soft":
                    feat_emb = self.dropout(feat_emb)
                
                elif self.mask_mode == "none":
                    pass
                else:
                    raise ValueError(f"Unknown mask mode: {self.mask_mode}")

            feat_inputs.append(feat_emb)

        concat_e = torch.cat([item_e] + feat_inputs, dim=-1) 

        encoded = self.encoder(concat_e) 

        predictions = {}
        for name in self.feat_names:
            predictions[name] = self.pred_heads[name](encoded)
        return predictions, feat_mask

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        feat_dict: Dict[str, torch.LongTensor],
        loss_mask,
        feat_mask: Dict[str, bool],
    ) -> torch.Tensor:
        total_loss = 0.0
        for name in self.feat_names:
            pred_emb = predictions[name]

            if not feat_mask[name]:
                # 防止没有参与损失而报错
                total_loss += 0 * pred_emb.sum()
                continue  
            
            if self.loss_type == "contrastive":
                # L2归一化
                pred_norm = F.normalize(pred_emb, dim=-1)
                table_norm  = F.normalize(self.feat_emb[name].weight, dim=-1)
                logits = torch.matmul(pred_norm, table_norm.T) / self.temperature
                logits = logits.reshape(-1, logits.size(-1))
                labels = feat_dict[name].reshape(-1)
                loss = F.cross_entropy(logits, labels, reduction='none')
                loss = loss[loss_mask].mean()
            elif self.loss_type == "mse":
                target_emb = self.feat_emb[name](feat_dict[name]) 
                loss = F.mse_loss(pred_emb, target_emb, reduction='none').mean(-1).reshape(-1)
                loss = loss[loss_mask].mean()
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            total_loss += loss
        return total_loss / len(self.feat_names)

    def training_step(
        self,
        item_id: torch.Tensor,
        feat_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        preds, feat_mask = self.forward(item_id, feat_dict)
        loss_mask = (item_id != 0).bool().reshape(-1)
        loss = self.compute_loss(preds, feat_dict, loss_mask, feat_mask)
        return loss
