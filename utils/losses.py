import torch
import torch.nn as nn
from .gather import GatherLayer
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, scale_factors=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # normalize the features
        features = F.normalize(features, dim=-1)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]            
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

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
        # scale the logits (cosine sim) by the scale factors (cross-modal sim)
        if scale_factors is not None:
            scale_factors = scale_factors.repeat(anchor_count, contrast_count)
            logits = logits * scale_factors

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

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class APheSCL(nn.Module):
    def __init__(self, world_size, batch_size, temperature=0.07):
        super().__init__()
        self.world_size = world_size
        self.batch_size = batch_size
        self.temperature = temperature
        self.contrastive_loss = SupConLoss(temperature)

    def compute_categorical_sim(self, cat_phenotypes):
        device = cat_phenotypes.device
        batch_size, n_features = cat_phenotypes.shape
        
        similarity = torch.zeros(batch_size, batch_size, device=device)
        
        for i in range(n_features):
            feature = cat_phenotypes[:, i]            
            valid_mask = (feature != -1)
            confidence = valid_mask.float().sum() / batch_size # confidence based on the ratio of valid values
            
            if valid_mask.sum() <= 1:
                # Skip this feature if there are not enough valid values
                continue
            
            # Compute similarity for this feature
            feature_sim = torch.zeros(batch_size, batch_size, device=device)
            
            # Get valid indices
            valid_indices = torch.where(valid_mask)[0]
            
            # Compute similarity only for valid pairs
            for idx1, i in enumerate(valid_indices):
                for idx2, j in enumerate(valid_indices):
                    if feature[i] == feature[j]:
                        feature_sim[i, j] = 1.0 * confidence
            
            # Update overall similarity
            similarity += feature_sim
        
        return similarity / n_features

    def compute_continuous_sim(self, cont_phenotypes):
        """
        Compute similarity for continuous phenotypes with careful handling of missing values
        """
        device = cont_phenotypes.device
        batch_size, n_features = cont_phenotypes.shape
        
        # Initialize similarity matrix
        similarity = torch.zeros(batch_size, batch_size, device=device)
        
        for i in range(n_features):
            feature = cont_phenotypes[:, i]            
            valid_mask = (feature != -1)
            confidence = valid_mask.float().sum() / batch_size
            
            if valid_mask.sum() <= 1:
                continue
                
            valid_values = feature[valid_mask]
            
            mean = valid_values.mean()
            std = valid_values.std() + 1e-6
            
            # Normalize valid values
            normalized = (valid_values - mean) / std
            
            # Compute pairwise distances for valid values
            dist_matrix = torch.zeros(batch_size, batch_size, device=device)
            valid_indices = torch.where(valid_mask)[0]
            
            for idx1, i in enumerate(valid_indices):
                for idx2, j in enumerate(valid_indices):
                    if i != j:
                        dist = torch.abs(normalized[idx1] - normalized[idx2])
                        dist_matrix[i, j] = dist
            
            # Convert distances to similarities using Gaussian kernel
            sigma = 1.0
            sim = torch.exp(-dist_matrix / (2 * sigma * sigma)) * confidence
            
            # Update overall similarity
            similarity += sim
        
        return similarity / n_features

    def compute_scale_factors(self, cat_phenotypes=None, cont_phenotypes=None):
        if cat_phenotypes is None and cont_phenotypes is None:
            return None
        bs = cat_phenotypes.shape[0] if cat_phenotypes is not None else cont_phenotypes.shape[0]
        device = cat_phenotypes.device if cat_phenotypes is not None else cont_phenotypes.device

        phenotype_sim = torch.ones((bs, bs), device=device) # base scale_factor = 1
        
        if cat_phenotypes is not None:
            cat_sim = self.compute_categorical_sim(cat_phenotypes)
            phenotype_sim += cat_sim
            
        if cont_phenotypes is not None:
            cont_sim = self.compute_continuous_sim(cont_phenotypes)
            phenotype_sim += cont_sim       
        
        return phenotype_sim

    def forward(self, features, labels=None, cat_phenotypes=None, cont_phenotypes=None):

        
        if self.world_size > 1:
            features = torch.cat(GatherLayer.apply(features), dim=0)
            labels = torch.cat(GatherLayer.apply(labels), dim=0) if labels is not None else None
            cat_phenotypes = torch.cat(GatherLayer.apply(cat_phenotypes), dim=0) if cat_phenotypes is not None else None
            cont_phenotypes = torch.cat(GatherLayer.apply(cont_phenotypes), dim=0) if cont_phenotypes is not None else None
        
        scale_factors = self.compute_scale_factors(cat_phenotypes, cont_phenotypes)
        
        loss = self.contrastive_loss(features, labels, scale_factors)
        return loss


class MultiTaskSampleRelationLoss(nn.Module):
    def __init__(self, batch_size, world_size, num_phenotype, mode='all'):
        super().__init__()
        self.batch_size = batch_size
        self.world_size = world_size
        self.num_phenotype = num_phenotype
        self.mode = mode
        self.sim = nn.CosineSimilarity(dim=-1)
        self.criteria = nn.MSELoss()

    def sample_relation(self, mat):
        # calculate the cosine similarity between samples
        # and mask-out the diagonal elements (self-similarity)
        # sim: (N, N)
        N = mat.shape[0]
        mask = torch.eye(N, dtype=torch.bool).to(mat.device)
        sim = self.sim(mat.unsqueeze(1), mat.unsqueeze(0))
        sim[mask] = 0
        return sim
    
    def forward(self, features, continuous_phenotypes):
        # features: (B * V, C)
        # continuous_phenotypes: (B, K) K: num_phenotype
        N = self.world_size * self.batch_size
        V = features.shape[0] // self.batch_size

        assert continuous_phenotypes.shape[1] == self.num_phenotype, "Number of phenotypes must match"
        K = continuous_phenotypes.shape[1]

        if self.world_size > 1:
            features = torch.cat(GatherLayer.apply(features), dim=0)
            cp = torch.cat(GatherLayer.apply(continuous_phenotypes), dim=0)
        
        loss = 0
        # supervision on one view
        if self.mode == 'one':
            
            features = rearrange(features, '(n v) c -> n v c', n=N, v=V)
            features = features.view(N, V, -1)[:, 0, :] # (N, C)

            for i in range(K):
                sub_phe = cp[:, i].view(-1, 1)
                valid_idx = (sub_phe != -1).view(-1)
                fea_sim = self.sample_relation(features[valid_idx])
                phe_sim = self.sample_relation(sub_phe[valid_idx])
                loss += self.criteria(fea_sim, phe_sim)

        # supervision on all views
        elif self.mode == 'all':
            features = features.view(N, V, -1).view(N * V, -1) # (N * V, C)
            cp = cp.unsqueeze(1).repeat_interleave(V, dim=1).view(N * V, K)
            for i in range(K):                
                sub_phe = cp[:, i].view(-1, 1)
                valid_idx = (sub_phe != -1).view(-1)
                fea_sim = self.sample_relation(features[valid_idx])
                phe_sim = self.sample_relation(sub_phe[valid_idx])
                loss += self.criteria(fea_sim, phe_sim)

        return loss / K


class ProbabilityLoss(nn.Module):
    def __init__(self, batch_size, world_size):
        super(ProbabilityLoss, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, logits_stu, logits_tch):
        assert logits_stu.size() == logits_tch.size()
        if self.world_size > 1:
            logits_stu = torch.cat(GatherLayer.apply(logits_stu), dim=0)
            logits_tch = torch.cat(GatherLayer.apply(logits_tch), dim=0)
        softmax1 = torch.log_softmax(logits_stu, dim=-1)
        softmax2 = nn.Softmax(dim=-1)(logits_tch)

        probability_loss = self.criterion(softmax1, softmax2)
        return probability_loss


class BatchLoss(nn.Module):
    def __init__(self, batch_size, world_size):
        super(BatchLoss, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, features_stu, features_tch):
        assert features_stu.size() == features_tch.size()
        N = self.batch_size * self.world_size
        # gather data from all the processes
        if self.world_size > 1:
            features_stu = torch.cat(GatherLayer.apply(features_stu), dim=0)
            features_tch = torch.cat(GatherLayer.apply(features_tch), dim=0)
        # reshape as N*C
        features_stu = features_stu.view(N, -1)
        features_tch = features_tch.view(N, -1)

        # form N*N similarity matrix
        sim_stu = features_stu.mm(features_stu.t())
        norm_stu = torch.norm(sim_stu, 2, 1).view(-1, 1)
        sim_stu = sim_stu / norm_stu

        sim_tch = features_tch.mm(features_tch.t())
        norm_tch = torch.norm(sim_tch, 2, 1).view(-1, 1)
        sim_tch = sim_tch / norm_tch

        batch_loss = (sim_stu - sim_tch) ** 2 / N
        batch_loss = batch_loss.sum()
        return batch_loss


class MultiviewCrossEntropy(nn.Module):
    def __init__(self, mode='all'):
        super().__init__()  # Properly initialize the base class
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mode = mode

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        # logits: [B, V, C]
        # labels: [B]
        B, V, C = logits.shape
        if self.mode == 'all':
            logits = logits.view(B * V, C)  # Reshape logits to merge batch and views
            labels = labels.unsqueeze(1).repeat(1, V).view(-1)  # Repeat labels for each view and flatten
        elif self.mode == 'one':
            logits = logits[:, 0, :]  # Use only the first view's logits
            # labels remain unchanged
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return self.ce(logits, labels)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.alpha = torch.Tensor([403./871 ,1-403./871])
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class MultiviewFocalLoss(nn.Module):
    def __init__(self, mode='all', gamma=2.0, alpha=None):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.mode = mode

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [B, V, C]
        # labels: [B]
        B, V, C = logits.shape
        if self.mode == 'all':
            logits = logits.view(B * V, C)  # Reshape logits to merge batch and views
            labels = labels.unsqueeze(1).repeat(1, V).view(-1)  # Repeat labels for each view and flatten
        elif self.mode == 'one':
            logits = logits[:, 0, :]  # Use only the first view's logits
            # labels remain unchanged
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return self.focal_loss(logits, labels)
