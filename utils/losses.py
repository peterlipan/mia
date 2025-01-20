import torch
import torch.nn as nn
from .gather import GatherLayer
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.eps = eps

    def forward(self, features, similarity_matrix):
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        
        # Tile similarity matrix
        similarity_matrix = similarity_matrix.repeat(anchor_count, contrast_count)
        
        # Compute logits with numerical stability
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        anchor_dot_contrast = torch.clamp(anchor_dot_contrast, min=-1.0, max=1.0)
        anchor_dot_contrast = anchor_dot_contrast / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(similarity_matrix),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        
        exp_logits = torch.exp(logits) * logits_mask
        
        # Compute log_prob with careful handling of zero denominators
        pos_mask = similarity_matrix
        neg_mask = (1 - similarity_matrix) * logits_mask
        
        pos_exp_sum = torch.sum(exp_logits * pos_mask, dim=1, keepdim=True)
        neg_exp_sum = torch.sum(exp_logits * neg_mask, dim=1, keepdim=True)
        
        # Add small epsilon to both sums
        denominator = pos_exp_sum + neg_exp_sum + self.eps
        
        log_prob = logits - torch.log(denominator)
        
        # Handle positive pairs carefully
        pos_mask_sum = pos_mask.sum(1)
        # Add eps to avoid division by zero
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask_sum + self.eps)
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss

class APheSCL(nn.Module):
    def __init__(self, world_size, batch_size, temperature=0.07, base_temperature=0.07, eps=1e-8):
        super().__init__()
        self.world_size = world_size
        self.batch_size = batch_size
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.eps = eps
        self.contrastive_loss = SupConLoss(temperature, base_temperature, eps)

    def compute_categorical_sim(self, cat_phenotypes):
        """
        Compute similarity for categorical phenotypes
        Args:
            cat_phenotypes: tensor of shape [batch_size, n_cat_features]
        Returns:
            similarity matrix of shape [batch_size, batch_size]
        """
        device = cat_phenotypes.device
        batch_size, n_features = cat_phenotypes.shape
        
        # Initialize similarity matrix
        similarity = torch.ones(batch_size, batch_size, device=device)
        
        for i in range(n_features):
            feature = cat_phenotypes[:, i]
            
            # Create mask for valid values (assuming -1 is used for missing values)
            valid_mask = (feature != -1)
            
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
                        feature_sim[i, j] = 1.0
            
            # Update overall similarity
            similarity *= (feature_sim + (1 - valid_mask.float().unsqueeze(1) * valid_mask.float().unsqueeze(0)))
        
        return similarity

    def compute_continuous_sim(self, cont_phenotypes):
        """
        Compute similarity for continuous phenotypes with careful handling of missing values
        """
        device = cont_phenotypes.device
        batch_size, n_features = cont_phenotypes.shape
        
        # Initialize similarity matrix
        similarity = torch.ones(batch_size, batch_size, device=device)
        
        for i in range(n_features):
            feature = cont_phenotypes[:, i]
            
            # Create mask for valid values
            valid_mask = (feature != -1)
            
            if valid_mask.sum() <= 1:
                # Skip this feature if there are not enough valid values
                continue
                
            valid_values = feature[valid_mask]
            
            # Compute mean and std only on valid values
            mean = valid_values.mean()
            # Add eps to avoid zero std
            std = valid_values.std() + 1e-6 if len(valid_values) > 1 else torch.tensor(1.0).to(device)
            
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
            sigma = 1.0  # You can adjust this parameter
            sim = torch.exp(-dist_matrix / (2 * sigma * sigma))
            
            # Update overall similarity
            similarity *= sim
        
        return similarity

    def compute_similarity_matrix(self, labels, cat_phenotypes=None, cont_phenotypes=None):
        device = labels.device
        batch_size = len(labels)
        
        # Create label mask
        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float().to(device)
        
        if cat_phenotypes is None and cont_phenotypes is None:
            return label_mask
        
        phenotype_sim = torch.ones((batch_size, batch_size), device=device)
        
        if cat_phenotypes is not None:
            cat_sim = self.compute_categorical_sim(cat_phenotypes)
            phenotype_sim *= cat_sim
            
        if cont_phenotypes is not None:
            cont_sim = self.compute_continuous_sim(cont_phenotypes)
            phenotype_sim *= cont_sim
        
        # Ensure similarity values are between 0 and 1
        phenotype_sim = torch.clamp(phenotype_sim, 0, 1)
        
        # Final similarity matrix
        final_sim = label_mask * phenotype_sim
        
        # Ensure each sample has at least one positive pair
        row_sums = final_sim.sum(dim=1)
        no_positives = row_sums == 0
        if no_positives.any():
            final_sim[no_positives] = label_mask[no_positives]
        
        return final_sim

    def forward(self, features, labels, cat_phenotypes=None, cont_phenotypes=None):
        # Input validation
        assert not torch.isnan(features).any(), "NaN in features"
        assert not torch.isinf(features).any(), "Inf in features"
        
        if self.world_size > 1:
            features = torch.cat(GatherLayer.apply(features), dim=0)
            labels = torch.cat(GatherLayer.apply(labels), dim=0)
            if cat_phenotypes is not None:
                cat_phenotypes = torch.cat(GatherLayer.apply(cat_phenotypes), dim=0)
            if cont_phenotypes is not None:
                cont_phenotypes = torch.cat(GatherLayer.apply(cont_phenotypes), dim=0)
        
        similarity_matrix = self.compute_similarity_matrix(labels, cat_phenotypes, cont_phenotypes)
        
        # Validate similarity matrix
        assert not torch.isnan(similarity_matrix).any(), "NaN in similarity matrix"
        assert not torch.isinf(similarity_matrix).any(), "Inf in similarity matrix"
        assert (similarity_matrix >= 0).all() and (similarity_matrix <= 1).all(), "Similarity values out of range"
        
        loss = self.contrastive_loss(features, similarity_matrix)
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
