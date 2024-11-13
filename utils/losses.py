import torch
import torch.nn as nn
from .gather import GatherLayer


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