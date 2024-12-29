import torch
import torch.nn as nn
from .gather import GatherLayer
import torch.nn.functional as F
from einops import rearrange


class GraphAttentionLoss(nn.Module):
    def __init__(self, batch_size, world_size):
        super().__init__()
        self.batch_size = batch_size
        self.world_size = world_size
    
    def forward(self, group_attn, rois):
        # group_attn: (B, 1, num_groups, num_rois)
        # rois: (B, num_rois, num_features)
        N = self.world_size * self.batch_size
        G, R = group_attn.shape[2], group_attn.shape[3]
        
        if self.world_size > 1:
            group_attn = torch.cat(GatherLayer.apply(group_attn), dim=0)
            rois = torch.cat(GatherLayer.apply(rois), dim=0)
        
        # Reshape group_attn to (N, R, G)
        group_attn = group_attn.squeeze(1).transpose(1, 2)  # Shape: (N, R, G)

        # Compute cosine similarity for group_attn
        attn_sim = torch.bmm(group_attn, group_attn.transpose(1, 2))  # Shape: (N, R, R)
        attn_norms = torch.norm(group_attn, p=2, dim=2, keepdim=True)  # Shape: (N, R, 1)
        attn_sim = attn_sim / (attn_norms @ attn_norms.transpose(1, 2))  # Shape: (N, R, R)

        # Compute cosine similarity for rois
        roi_sim = torch.bmm(rois, rois.transpose(1, 2))  # Shape: (N, R, R)
        roi_norms = torch.norm(rois, p=2, dim=2, keepdim=True)  # Shape: (N, R, 1)
        roi_sim = roi_sim / (roi_norms @ roi_norms.transpose(1, 2))  # Shape: (N, R, R)

        # Handle division by zero for roi_sim
        roi_sim[torch.isnan(roi_sim)] = 0  # Set NaNs to 0 (if both norms are zero)

        # Calculate loss
        loss = nn.MSELoss()(attn_sim, roi_sim)
        
        return loss


class SoftContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for the supervised loss

    def forward(self, features, adj):
        # features: (B, n_views, C)
        # adj: (B, B)

        device = features.device

        # Reshape features to [B * n_views, C]
        batch_size, n_views, C = features.shape
        contrast_feature = features.view(-1, C)  # Flatten to [B * n_views, C]

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(adj),
            1,
            torch.arange(batch_size * n_views).view(-1, 1).to(device),
            0
        )

        # Unsupervised Loss
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        unsupervised_loss = -log_prob.mean()

        # Supervised Loss
        # Create a mask for valid phenotypic similarities (only positive pairs)
        positive_mask = (adj > 0).float()
        weighted_mask = adj * logits_mask * positive_mask
        
        # Compute mean of log-likelihood over positive pairs only
        mask_pos_pairs = weighted_mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # Avoid division by zero
        mean_log_prob_pos = (weighted_mask * log_prob).sum(1) / mask_pos_pairs

        # Supervised loss
        supervised_loss = - (self.temperature / 1.0) * mean_log_prob_pos
        supervised_loss = supervised_loss.view(n_views, batch_size).mean()

        # Combine losses
        loss = self.alpha * supervised_loss + (1 - self.alpha) * unsupervised_loss

        return loss


class SupConLoss(nn.Module):
    
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            adj: Adjacent matrix of the samples, [bsz, bsz], 0<= mask_{i,j} <=1 
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        B, V, C = features.shape

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class MultiTaskSupervisedContrast(nn.Module):
    def __init__(self, batch_size, world_size, num_phenotype, temperature=0.07):
        super().__init__()
        self.batch_size = batch_size
        self.world_size = world_size
        self.num_phenotype = num_phenotype
        self.contrastive_loss = SupConLoss(temperature=temperature)
        self.sim = nn.CosineSimilarity(dim=2)
    
    def forward(self, features, phenotypes=None, labels=None, weights=None):
        # features: (B, n_views, C)
        # phenotypes: (B, K) K: Number of category phenotypes
        N = self.world_size * self.batch_size

        if self.world_size > 1:
            features = torch.cat(GatherLayer.apply(features), dim=0)
            phenotypes = torch.cat(GatherLayer.apply(phenotypes), dim=0) if phenotypes is not None else None
            labels = torch.cat(GatherLayer.apply(labels), dim=0) if labels is not None else None
        
        # chunk the features for each task
        _, _, C = features.shape
        assert C % self.num_phenotype == 0, "Number of features must be divisible by the number of phenotypes"
        features = features.chunk(self.num_phenotype, dim=-1) # Tuple of (B, n_views, C/K)
        
        loss = self.contrastive_loss(features[0], labels)
        if phenotypes is not None:
            for i in range(1, self.num_phenotype):
                sub_phe = phenotypes[:, i - 1].view(-1)
                valid_idx = sub_phe != -1
                loss += self.contrastive_loss(features[i][valid_idx], sub_phe[valid_idx])

        return loss
