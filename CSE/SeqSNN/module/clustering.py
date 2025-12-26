'''
Module: clustering.py
Modified by: Hyunwoo Kang
Last Modified: 2025-10-08 16:57
Changes: Attention_cluster_assigner, Cluster_wise_linear, similarity matrix 및 loss 함수 추가
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt

class CSEv1(nn.Module):
    def __init__(self, method_type='attention', n_cluster=3, d_model=64, device=0, **kwargs):
        super(CSEv1, self).__init__()
        self.method_type = method_type
        self.n_cluster = n_cluster
        self.d_model = d_model
        self.device = device
        self.assigner = None

        if method_type == 'attention':
            self.assigner = Attention_cluster_assigner(
                n_vars=kwargs.get('n_vars'),
                n_cluster=n_cluster,
                seq_len=kwargs.get('seq_len'),
                d_model=d_model,
            )
        elif method_type == 'kmeans':
            self.assigner = KMeans_cluster_assigner(
                n_vars=kwargs.get('n_vars'),
                n_cluster=n_cluster,
                seq_len=kwargs.get('seq_len'),
                d_model=d_model,
                device=device,
            )
        self.to(device)

    def forward(self, inputs):
        self.cluster_prob, self.cluster_emb, self.x_emb = self.assigner(inputs, self.assigner.cluster_emb if self.method_type=='attention' else self.assigner.centroids)
        self.clustering_encoding = self.cluster_base_encoding(self.cluster_prob, inputs.size(1))
        return self.clustering_encoding

    def cluster_base_encoding(self, cluster_prob, seq_len):
        cluster_prob_soft = cluster_prob.permute(2,0,1).unsqueeze(-1)
        cluster_prob_bern = torch.bernoulli(cluster_prob_soft) # K, B, N
        cluster_prob_extend = cluster_prob_bern.repeat(1,1,1,seq_len)
        cluster_prob_hard = cluster_prob_soft + (cluster_prob_extend - cluster_prob_soft).detach()
        return cluster_prob_hard  #[n_cluster, n_vars, seq_len, B]

class Attention_cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device='cuda', epsilon=0.05):
        super(Attention_cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.d_model = d_model
        self.epsilon = epsilon
        self.device = device
        self.linear = nn.Linear(seq_len, d_model)
        
        # Cluster embeddings
        self.cluster_emb = nn.Parameter(torch.empty(self.n_cluster, self.d_model).to(device))
        nn.init.kaiming_uniform_(self.cluster_emb, a=math.sqrt(5))
        
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.p2c = CrossAttention(d_model, n_heads=1)
        self.i = 0

    def forward(self, x, cluster_emb, return_type='individual'):
        # x: [bs, seq_len, n_vars]
        # cluster_emb: [n_cluster, d_model]
        n_vars = x.shape[-1]
        x = x.permute(0, 2, 1)
        x_emb = self.linear(x).reshape(-1, self.d_model)  # [bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn / n_vars), 1)
        prob = torch.mm(self.l2norm(x_emb), self.l2norm(cluster_emb).t()).reshape(bs, n_vars, self.n_cluster)
        # prob: [bs, n_vars, n_cluster]
        prob_temp = prob.reshape(-1, self.n_cluster)
        prob_temp = sinkhorn(prob_temp, epsilon=self.epsilon)  # [n_vars*bs, n_cluster]
        prob_avg = torch.mean(prob, dim=0)  # [n_vars, n_cluster]
        prob_avg = sinkhorn(prob_avg, epsilon=self.epsilon)
        mask = self.concrete_bern(prob_avg)  # [bs, n_vars, n_cluster]

        x_emb_ = x_emb.reshape(bs, n_vars, -1)
        cluster_emb_ = cluster_emb.repeat(bs, 1, 1)
        cluster_emb_out = self.p2c(cluster_emb_, x_emb_, x_emb_, mask=mask.transpose(0, 1))
        cluster_emb_avg = torch.mean(cluster_emb_out, dim=0)

        # Return variable names unified: prob, cluster_emb, x_emb
        if return_type == 'individual':
            return prob_temp.reshape(prob.shape), cluster_emb_avg, x_emb
        elif return_type == 'average':
            return prob_avg, cluster_emb_avg, x_emb
     
    def concrete_bern(self, prob, temp = 0.07):
        random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
        prob_bern = ((prob + random_noise) / temp).sigmoid()
        return prob_bern
    
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):   #[n_vars, n_cluster]
    Q = torch.exp(out / epsilon)
    sum_Q = torch.sum(Q, dim=1, keepdim=True) 
    Q = Q / (sum_Q)
    return Q

class CrossAttention(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    input:
        queries: (bs, L, d_model)
        keys: (_, S, d_model)
        values: (bs, S, d_model)
        mask: (L, S)
    return: (bs, L, d_model)

    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1):
        super(CrossAttention, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = MaskAttention(scale=None, attention_dropout = dropout)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, mask=None):
        # input dim: d_model
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)
       
        out = self.inner_attention(
            queries,
            keys,
            values,
            mask,
        )

        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)
       

        return out # B, L, d_model
    
class MaskAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(MaskAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
    
        
        # scores = scores if mask == None else scores * mask
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
    
        A = A if mask == None else A * mask
        V = torch.einsum("bhls,bshd->blhd", A, values)
    
        return V.contiguous()

class Cluster_wise_linear(nn.Module):
    def __init__(self, n_cluster, n_vars, in_dim, out_dim, device='cuda'):
        super().__init__()
        self.n_cluster = n_cluster
        self.n_vars = n_vars
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linears = nn.ModuleList()
        for i in range(n_cluster):
            self.linears.append(nn.Linear(in_dim, out_dim))

        
    def forward(self, x, prob):
        # x: [bs, n_vars, in_dim]
        # prob: [n_vars, n_cluster]
        # return: [bs, n_vars, out_dim]
        bsz = x.shape[0]
        output = []
        for layer in self.linears:
            output.append(layer(x))
        output = torch.stack(output, dim=-1).to(x.device)  #[bsz, n_vars,  out_dim, n_cluster]
        prob = prob.unsqueeze(-1)  #[n_vars, n_cluster, 1]
        output = torch.matmul(output, prob).reshape(bsz, -1, self.out_dim)   #[bsz, n_vars, out_dim]
        return output
    
def get_similarity_matrix_update(batch_data, sigma=5.0):
        """
        Compute the similarity matrix between different channels of a time series in a batch.
        The similarity is computed using the exponential function on the squared Euclidean distance
        between mean temporal differences of channels.
        
        Parameters:
            batch_data (torch.Tensor): Input data of shape (batch_len, seq_len, channel).
            sigma (float): Parameter controlling the spread of the Gaussian similarity function.
            
        Returns:
            torch.Tensor: Similarity matrix of shape (channel, channel).
        """
        batch_len, seq_len, num_channels = batch_data.shape
        similarity_matrix = torch.zeros((num_channels, num_channels), device=batch_data.device)

        # Compute point-by-point differences along the sequence length
        time_diffs = batch_data[:, 1:, :] - batch_data[:, :-1, :]  # Shape: (batch_len, seq_len-1, channel)
        
        # Compute mean of these differences over batch and sequence length
        channel_representations = time_diffs.mean(dim=(0, 1))  # Shape: (channel,)
        
        # Compute pairwise similarity
        '''
        for i in range(num_channels):
            for j in range(num_channels):
                diff = torch.norm(channel_representations[i] - channel_representations[j]) ** 2
                similarity_matrix[i, j] = torch.exp(-diff / (2 * sigma ** 2))
        '''
        # Vectorized computation of pairwise similarity
        # Expand dimensions for broadcasting
        repr_i = channel_representations.unsqueeze(1)  # (num_channels, 1)
        repr_j = channel_representations.unsqueeze(0)  # (1, num_channels)
        
        # Compute squared differences for all pairs at once
        diff_squared = (repr_i - repr_j) ** 2  # (num_channels, num_channels)
        
        # Apply exponential similarity function
        similarity_matrix = torch.exp(-diff_squared / (2 * sigma ** 2))
        

        return similarity_matrix.to(batch_data.device)

def similarity_loss_batch(prob, simMatrix):
        def concrete_bern(prob, temp = 0.07):
            random_noise = torch.empty_like(prob).uniform_(1e-10, 1 - 1e-10).to(prob.device)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            prob = torch.log(prob + 1e-10) - torch.log(1.0 - prob + 1e-10)
            prob_bern = ((prob + random_noise) / temp).sigmoid()
            return prob_bern
        if prob.dim() == 3:
            prob = prob.mean(dim=0) # set to [n_vars, n_cluster]
        membership = concrete_bern(prob)  #[n_vars, n_clusters]
        temp_1 = torch.mm(membership.t(), simMatrix) 
        SAS = torch.mm(temp_1, membership)
        _SS = 1 - torch.mm(membership, membership.t())
        loss = -torch.trace(SAS) + torch.trace(torch.mm(_SS, simMatrix)) + membership.shape[0]
        ent_loss = (-prob * torch.log(prob + 1e-15)).sum(dim=-1).mean()
        return loss + ent_loss

class KMeans_cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device='cpu', max_iter=10, tol=1e-4):
        super(KMeans_cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.seq_len = seq_len
        self.d_model = d_model
        self.device = device
        self.max_iter = max_iter
        self.tol = tol

        # 각 채널을 임베딩 하기 위한 선형 레이어 선언
        self.linear = nn.Linear(seq_len, d_model)

        # KMeans 클러스터 중심 초기화 (학습 가능한 파라미터)
        self.centroids = nn.Parameter(torch.randn(n_cluster, d_model).to(device))
        nn.init.kaiming_uniform_(self.centroids, a=math.sqrt(5))

    def kmeans_clustering(self, x_emb, centroids):
        '''
        Torch 기반 KMeans 클러스터링
        x_emb: [N, d_model] - 임베딩된 채널들
        centroids: [n_cluster, d_model] - 클러스터 중심
        return: cluster_assignments [N, n_cluster] - 각 샘플의 클러스터 할당 확률 (one-hot)
        '''
        for _ in range(self.max_iter):
            # 각 샘플과 모든 centroid 간의 거리 계산
            distances = torch.cdist(x_emb, centroids, p=2)  # [N, n_cluster]
            
            # 가장 가까운 클러스터에 할당
            cluster_ids = torch.argmin(distances, dim=1)  # [N]
            
            # 새로운 centroid 계산
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_cluster):
                mask = (cluster_ids == k)
                if mask.sum() > 0:
                    new_centroids[k] = x_emb[mask].mean(dim=0)
                else:
                    # 클러스터에 할당된 샘플이 없으면 기존 centroid 유지
                    new_centroids[k] = centroids[k]
            
            # 수렴 확인
            centroid_shift = torch.norm(new_centroids - centroids)
            centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        # One-hot 인코딩으로 클러스터 할당 변환
        cluster_assignments = F.one_hot(cluster_ids, num_classes=self.n_cluster).float()
        
        return cluster_assignments, centroids

    def forward(self, x, cluster_emb=None, return_type='individual'):
        '''
        x: [bs, seq_len, n_vars]
        return_type: 'individual' or 'average'
        return: 
            - prob: [bs, n_vars, n_cluster] if individual, [n_vars, n_cluster] if average
            - cluster_emb: [n_cluster, d_model]
            - x_emb: [bs*n_vars, d_model]
        '''
        n_vars = x.shape[-1]
        x = x.permute(0, 2, 1)  # [bs, n_vars, seq_len]
        x_emb = self.linear(x).reshape(-1, self.d_model)  # [bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn / n_vars), 1)

        # KMeans 클러스터링 수행
        cluster_assignments, updated_centroids = self.kmeans_clustering(x_emb, self.centroids)

        # 클러스터 할당 결과를 확률 형태로 변환 [bs*n_vars, n_cluster]
        prob = cluster_assignments.reshape(bs, n_vars, self.n_cluster)  # [bs, n_vars, n_cluster]

        # Return variable names unified: prob, cluster_emb, x_emb
        if return_type == 'individual':
            return prob, updated_centroids, x_emb
        elif return_type == 'average':
            prob_avg = prob.mean(dim=0)  # [n_vars, n_cluster]
            return prob_avg, updated_centroids, x_emb