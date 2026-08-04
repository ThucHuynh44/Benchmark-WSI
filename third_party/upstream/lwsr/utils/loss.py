import torch
import torch.nn.functional as F
from utils.conf import global_config_data

PAIR_LOSS_WEIGHT, CE_LOSS_WEIGHT, DC_LOSS_WEIGHT = global_config_data['pair_loss_weight'], global_config_data['ce_loss_weight'], global_config_data['dc_loss_weight']

class RetrievalLoss():
    def __init__(self):
        self.pred = None
        self.label = None
        self.features = None
        self.task = None
        self.previous_dist_mat = None
        self.current_dist_mat = None
        self.return_idx = None
        self.hash_weights = None
        self.is_ours = False

    def set(self, pred, label, features, task, is_ours, previous_dist_mat=None, current_dist_mat=None, return_idx=None,
            hash_weights=None):
        self.pred = pred
        self.label = label
        self.features = features
        self.hash_weights = hash_weights
        self.task = task
        self.is_ours = is_ours
        self.previous_dist_mat = previous_dist_mat
        self.current_dist_mat = current_dist_mat
        self.return_idx = return_idx

    def pair_loss_func(self):
        num = self.features.shape[0]
        feature_num = self.features.shape[1]
        label = self.label.unsqueeze(0).float()
        label = F.one_hot(label, num_classes=8).float()
        w_label = 2 * torch.chain_matmul(label, torch.t(label)) - 1
        Y = torch.chain_matmul(self.features, torch.t(self.features)) / feature_num - w_label
        rhl = torch.sum(torch.abs(torch.abs(self.features) - 1)) / num / feature_num

        if num == 1:
            loss = torch.sum(Y * Y / num / (num)) + 0.3 * rhl
        else:
            loss = torch.sum(Y * Y / num / (num - 1)) + 0.3 * rhl

        return loss

    def hash_loss_func(self):
        hash_bits = self.features.size()[1]
        cross_sim_mat = torch.matmul(self.features, self.features.T) / hash_bits
        cross_label_mat = torch.matmul(self.label, self.label.T)
        cross_label_mat[cross_label_mat < 1] = -1
        loss = (cross_sim_mat - cross_label_mat).pow(2).mean()
        return loss

    def cls_loss_func(self):
        return F.cross_entropy(self.pred, self.label)

    def dist_consistency_loss_func(self):
        original_distances = self.previous_dist_mat[self.return_idx][:, self.return_idx]
        new_distances = self.current_dist_mat

        mse = torch.mean((new_distances - original_distances) ** 2)

        return mse

    def moco_loss_func(self, temperature=0.1):
        loss = None
        counter = 0

        for target_label in range(self.task * 2, self.task * 2 + 1):
            pos_mask = self.label == target_label
            neg_mask = ~pos_mask
            query = self.features[pos_mask]
            key = self.features[pos_mask]
            queue = self.features[neg_mask]

            query_norm = F.normalize(query, dim=-1)
            key_norm = F.normalize(key, dim=-1)
            queue_norm = F.normalize(queue, dim=-1)
            pos = torch.einsum("nc,nc->n", [query_norm, key_norm]).unsqueeze(-1)
            neg = torch.einsum("nc,kc->nk", [query_norm, queue_norm])
            logits = torch.cat([pos, neg], dim=1)
            logits /= temperature
            targets = torch.zeros(query.size(0), device=query.device, dtype=torch.long)
            moco_loss = F.cross_entropy(logits, targets)

            if counter == 0:
                loss = moco_loss
            else:
                loss += moco_loss

            counter += 1

        return loss
    
    def normal_loss_func(self):
        return PAIR_LOSS_WEIGHT * self.pair_loss_func() + CE_LOSS_WEIGHT * self.cls_loss_func()
    
    def dcl_loss_func(self):
        return DC_LOSS_WEIGHT * self.dist_consistency_loss_func()
    
    def all_loss_func(self):
        return self.normal_loss_func() + self.dcl_loss_func()

    def total_loss_func(self):
        if self.task == 0 or self.previous_dist_mat is None or not self.is_ours:
            return PAIR_LOSS_WEIGHT * self.pair_loss_func() + CE_LOSS_WEIGHT * self.cls_loss_func()
        else:
            return PAIR_LOSS_WEIGHT * self.pair_loss_func() + CE_LOSS_WEIGHT * self.cls_loss_func() + DC_LOSS_WEIGHT * self.dist_consistency_loss_func()


class RetrievalLoss_WO_DCL():
    def __init__(self):
        self.pred = None
        self.label = None
        self.features = None
        self.task = None
        self.previous_dist_mat = None
        self.current_dist_mat = None
        self.return_idx = None
        self.hash_weights = None
        self.is_ours = False

    def set(self, pred, label, features, task, is_ours, previous_dist_mat=None, current_dist_mat=None, return_idx=None,
            hash_weights=None):
        self.pred = pred
        self.label = label
        self.features = features
        self.hash_weights = hash_weights
        self.task = task
        self.is_ours = is_ours
        self.previous_dist_mat = previous_dist_mat
        self.current_dist_mat = current_dist_mat
        self.return_idx = return_idx

    def pair_loss_func(self):
        num = self.features.shape[0]
        feature_num = self.features.shape[1]
        label = self.label.unsqueeze(0).float()
        label = F.one_hot(label, num_classes=8).float()
        w_label = 2 * torch.chain_matmul(label, torch.t(label)) - 1
        Y = torch.chain_matmul(self.features, torch.t(self.features)) / feature_num - w_label
        rhl = torch.sum(torch.abs(torch.abs(self.features) - 1)) / num / feature_num

        if num == 1:
            loss = torch.sum(Y * Y / num / (num)) + 0.3 * rhl
        else:
            loss = torch.sum(Y * Y / num / (num - 1)) + 0.3 * rhl

        return loss

    def hash_loss_func(self):
        hash_bits = self.features.size()[1]
        cross_sim_mat = torch.matmul(self.features, self.features.T) / hash_bits
        cross_label_mat = torch.matmul(self.label, self.label.T)
        cross_label_mat[cross_label_mat < 1] = -1
        loss = (cross_sim_mat - cross_label_mat).pow(2).mean()
        return loss

    def cls_loss_func(self):
        return F.cross_entropy(self.pred, self.label)

    def dist_consistency_loss_func(self):
        original_distances = self.previous_dist_mat[self.return_idx][:, self.return_idx]
        new_distances = self.current_dist_mat
        mse = torch.mean((new_distances - original_distances) ** 2)

        return mse

    def moco_loss_func(self, temperature=0.1):
        loss = None
        counter = 0

        for target_label in range(self.task * 2, self.task * 2 + 1):
            pos_mask = self.label == target_label
            neg_mask = ~pos_mask
            query = self.features[pos_mask]
            key = self.features[pos_mask]
            queue = self.features[neg_mask]

            query_norm = F.normalize(query, dim=-1)
            key_norm = F.normalize(key, dim=-1)
            queue_norm = F.normalize(queue, dim=-1)
            pos = torch.einsum("nc,nc->n", [query_norm, key_norm]).unsqueeze(-1)
            neg = torch.einsum("nc,kc->nk", [query_norm, queue_norm])
            logits = torch.cat([pos, neg], dim=1)
            logits /= temperature
            targets = torch.zeros(query.size(0), device=query.device, dtype=torch.long)
            moco_loss = F.cross_entropy(logits, targets)

            if counter == 0:
                loss = moco_loss
            else:
                loss += moco_loss

            counter += 1

        return loss

    def total_loss_func(self):
        if self.task == 0 or self.previous_dist_mat is None or not self.is_ours:
            return PAIR_LOSS_WEIGHT * self.pair_loss_func() + CE_LOSS_WEIGHT * self.cls_loss_func()
        else:
            return PAIR_LOSS_WEIGHT * self.pair_loss_func() + CE_LOSS_WEIGHT * self.cls_loss_func()
