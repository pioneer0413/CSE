'''
Module: clustering.py
Modified by: Hyunwoo Kang
Last Modified: 2025-10-08 16:57
Changes: Cluster_assigner, Cluster_wise_linear, similarity matrix 및 loss 함수 추가
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt

def cluster_base_encoding(cluster_prob, seq_len):
    cluster_prob_soft = cluster_prob.permute(2,0,1).unsqueeze(-1)
    cluster_prob_bern = torch.bernoulli(cluster_prob_soft) # K, B, N
    cluster_prob_extend = cluster_prob_bern.repeat(1,1,1,seq_len)
    cluster_prob_hard = cluster_prob_soft + (cluster_prob_extend - cluster_prob_soft).detach()
    return cluster_prob_hard  #[n_cluster, n_vars, seq_len, 1]

class Cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device='cuda', epsilon=0.05):
        super(Cluster_assigner, self).__init__()
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
        x = x.permute(0,2,1)
        x_emb = self.linear(x).reshape(-1, self.d_model)      #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn/n_vars), 1) 
        prob = torch.mm(self.l2norm(x_emb), self.l2norm(cluster_emb).t()).reshape(bs, n_vars, self.n_cluster)
        # prob: [bs, n_vars, n_cluster]
        prob_temp = prob.reshape(-1, self.n_cluster)
        prob_temp = sinkhorn(prob_temp, epsilon=self.epsilon) # [n_vars*bs, n_cluster]
        prob_avg = torch.mean(prob, dim=0)    #[n_vars, n_cluster]
        prob_avg = sinkhorn(prob_avg, epsilon=self.epsilon)
        mask = self.concrete_bern(prob_avg)   #[bs, n_vars, n_cluster]

        x_emb_ = x_emb.reshape(bs, n_vars,-1)
        cluster_emb_ = cluster_emb.repeat(bs,1,1)
        cluster_emb = self.p2c(cluster_emb_, x_emb_, x_emb_, mask=mask.transpose(0,1))
        cluster_emb_avg = torch.mean(cluster_emb, dim=0)
        #print(cluster_emb.shape, cluster_emb_.shape, x_emb_.shape, mask.shape)

        #print(f'prob_avg: {prob_avg.shape}, prob_temp: {prob_temp.shape}, prob: {prob.shape}')
    
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

    def forward(self, x, return_type='individual'):
        '''
        x: [bs, seq_len, n_vars]
        return_type: 'individual' or 'average'
        return: 
            - prob: [bs, n_vars, n_cluster] if individual, [n_vars, n_cluster] if average
            - centroids: [n_cluster, d_model]
            - x_emb: [bs*n_vars, d_model]
        '''
        n_vars = x.shape[-1]
        x = x.permute(0,2,1)  #[bs, n_vars, seq_len]
        x_emb = self.linear(x).reshape(-1, self.d_model)  #[bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn/n_vars), 1)

        # KMeans 클러스터링 수행
        cluster_assignments, updated_centroids = self.kmeans_clustering(x_emb, self.centroids)
        
        # 클러스터 할당 결과를 확률 형태로 변환 [bs*n_vars, n_cluster]
        prob = cluster_assignments.reshape(bs, n_vars, self.n_cluster)  # [bs, n_vars, n_cluster]

        if return_type == 'individual':
            # 각 배치의 개별 클러스터 할당 반환
            return prob, updated_centroids, x_emb
        elif return_type == 'average':
            # 배치 전체의 평균 클러스터 할당 반환
            prob_avg = prob.mean(dim=0)  # [n_vars, n_cluster]
            return prob_avg, updated_centroids, x_emb


    
class DBSCAN_cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device='cpu', eps=0.5, min_samples=5):
        super(DBSCAN_cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster  # 최대 클러스터 수 (실제로는 DBSCAN이 자동으로 결정)
        self.seq_len = seq_len
        self.d_model = d_model
        self.device = device
        self.eps = eps  # 이웃으로 간주할 최대 거리
        self.min_samples = min_samples  # 핵심 포인트가 되기 위한 최소 이웃 수
        
        # 각 채널을 임베딩 하기 위한 선형 레이어 선언
        self.linear = nn.Linear(seq_len, d_model)
        
        # DBSCAN은 centroid가 없지만, 호환성을 위해 더미 파라미터 생성
        self.dummy_centroids = nn.Parameter(torch.randn(n_cluster, d_model).to(device), requires_grad=False)

    def dbscan_clustering(self, x_emb):
        '''
        완전 벡터화된 Torch 기반 DBSCAN 클러스터링 (최대 GPU 활용)
        x_emb: [N, d_model] - 임베딩된 채널들
        return: 
            - cluster_assignments [N, n_cluster] - 각 샘플의 클러스터 할당 (one-hot)
            - cluster_centers [n_cluster, d_model] - 각 클러스터의 중심
        '''
        N = x_emb.shape[0]
        device = x_emb.device
        
        # 모든 포인트 간의 거리 행렬 계산 (한 번만)
        distances = torch.cdist(x_emb, x_emb, p=2)  # [N, N]
        
        # 각 포인트의 이웃 찾기 (자기 자신 제외) - 완전 벡터화
        neighbors = (distances <= self.eps) & (~torch.eye(N, dtype=torch.bool, device=device))
        neighbor_counts = neighbors.sum(dim=1)  # 각 포인트의 이웃 수
        
        # 핵심 포인트 찾기 (이웃이 min_samples 이상)
        core_points = neighbor_counts >= self.min_samples
        
        # 연결 성분 찾기 (Union-Find를 행렬 연산으로 근사)
        # 핵심 포인트들 간의 연결성을 계산
        core_indices = torch.where(core_points)[0]
        
        if len(core_indices) == 0:
            # 핵심 포인트가 없으면 모두 노이즈
            cluster_labels = torch.zeros(N, dtype=torch.long, device=device)
            num_clusters_found = 1
        else:
            # 핵심 포인트들 간의 연결 그래프 구축
            core_neighbors = neighbors[core_indices][:, core_indices]
            
            # 전이적 폐포(transitive closure) 계산으로 연결 성분 찾기
            # 반복적으로 연결성을 확장
            connectivity = core_neighbors.clone().float()
            for _ in range(min(10, len(core_indices))):  # 최대 10번 반복으로 제한
                connectivity = torch.mm(connectivity, connectivity).clamp(0, 1)
            
            # 연결 성분에 클러스터 ID 할당
            visited = torch.zeros(len(core_indices), dtype=torch.bool, device=device)
            cluster_labels = torch.full((N,), -1, dtype=torch.long, device=device)
            current_cluster = 0
            
            for i in range(len(core_indices)):
                if visited[i]:
                    continue
                
                # i와 연결된 모든 핵심 포인트 찾기
                component = connectivity[i] > 0
                visited |= component
                
                # 이 연결 성분의 모든 핵심 포인트
                component_cores = core_indices[component]
                
                # 핵심 포인트들에 클러스터 ID 할당
                cluster_labels[component_cores] = current_cluster
                
                # 핵심 포인트들의 이웃(경계 포인트)도 같은 클러스터에 할당
                component_neighbors = neighbors[component_cores].any(dim=0)
                cluster_labels[component_neighbors] = current_cluster
                
                current_cluster += 1
            
            num_clusters_found = current_cluster
        
        # 노이즈 포인트들을 가장 가까운 클러스터에 할당 (선택적)
        noise_mask = cluster_labels == -1
        if noise_mask.any():
            noise_indices = torch.where(noise_mask)[0]
            clustered_indices = torch.where(~noise_mask)[0]
            
            if len(clustered_indices) > 0:
                # 노이즈 포인트와 클러스터된 포인트 간 거리
                noise_to_clustered_dist = distances[noise_indices][:, clustered_indices]
                # 가장 가까운 클러스터된 포인트 찾기
                nearest_clustered = clustered_indices[noise_to_clustered_dist.argmin(dim=1)]
                # 해당 클러스터에 할당
                cluster_labels[noise_indices] = cluster_labels[nearest_clustered]
        
        # n_cluster 맞추기 (패딩 또는 병합)
        if num_clusters_found > self.n_cluster:
            # 클러스터가 너무 많으면 작은 클러스터들을 병합
            unique_labels, counts = torch.unique(cluster_labels, return_counts=True)
            # 가장 큰 n_cluster 개만 유지
            _, top_k_indices = torch.topk(counts, min(self.n_cluster, len(counts)))
            top_k_labels = unique_labels[top_k_indices]
            
            # 작은 클러스터들을 가장 가까운 큰 클러스터에 병합
            new_labels = torch.full_like(cluster_labels, -1)
            for new_id, old_label in enumerate(top_k_labels):
                new_labels[cluster_labels == old_label] = new_id
            
            # 병합되지 않은 포인트들을 가장 가까운 클러스터에 할당
            unassigned_mask = new_labels == -1
            if unassigned_mask.any():
                unassigned_indices = torch.where(unassigned_mask)[0]
                assigned_indices = torch.where(~unassigned_mask)[0]
                
                unassigned_to_assigned_dist = distances[unassigned_indices][:, assigned_indices]
                nearest_assigned = assigned_indices[unassigned_to_assigned_dist.argmin(dim=1)]
                new_labels[unassigned_indices] = new_labels[nearest_assigned]
            
            cluster_labels = new_labels
            num_clusters_found = self.n_cluster
        
        # One-hot 인코딩 (n_cluster 크기로 패딩)
        if num_clusters_found < self.n_cluster:
            # 부족한 클러스터는 빈 클러스터로 남김
            cluster_assignments = torch.zeros(N, self.n_cluster, device=x_emb.device)
            cluster_assignments.scatter_(1, cluster_labels.unsqueeze(1), 1.0)
        else:
            cluster_assignments = F.one_hot(cluster_labels, num_classes=self.n_cluster).float()
        
        # 각 클러스터의 중심 계산 (완전 벡터화)
        cluster_centers = torch.zeros(self.n_cluster, self.d_model, device=device)
        if num_clusters_found > 0:
            # One-hot 인코딩을 이용한 벡터화된 중심 계산
            # cluster_labels를 one-hot으로 변환
            labels_onehot = F.one_hot(cluster_labels.clamp(min=0), num_classes=max(num_clusters_found, 1)).float()  # [N, num_clusters]
            # 각 클러스터의 포인트 수
            cluster_sizes = labels_onehot.sum(dim=0, keepdim=True).T  # [num_clusters, 1]
            # 각 클러스터의 중심 계산 (행렬 곱으로)
            cluster_centers[:num_clusters_found] = torch.mm(labels_onehot.T, x_emb) / (cluster_sizes + 1e-10)
        
        # 빈 클러스터는 랜덤 포인트로 초기화 (벡터화)
        if num_clusters_found < self.n_cluster:
            random_indices = torch.randint(0, N, (self.n_cluster - num_clusters_found,), device=device)
            cluster_centers[num_clusters_found:] = x_emb[random_indices]
        
        return cluster_assignments, cluster_centers

    def forward(self, x, return_type='individual'):
        '''
        x: [bs, seq_len, n_vars]
        return_type: 'individual' or 'average'
        return: 
            - prob: [bs, n_vars, n_cluster] if individual, [n_vars, n_cluster] if average
            - cluster_centers: [n_cluster, d_model]
            - x_emb: [bs*n_vars, d_model]
        '''
        n_vars = x.shape[-1]
        x = x.permute(0, 2, 1)  # [bs, n_vars, seq_len]
        x_emb = self.linear(x).reshape(-1, self.d_model)  # [bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn / n_vars), 1)

        # DBSCAN 클러스터링 수행
        cluster_assignments, cluster_centers = self.dbscan_clustering(x_emb)
        
        # 클러스터 할당 결과를 reshape
        prob = cluster_assignments.reshape(bs, n_vars, self.n_cluster)  # [bs, n_vars, n_cluster]

        if return_type == 'individual':
            # 각 배치의 개별 클러스터 할당 반환
            return prob, cluster_centers, x_emb
        elif return_type == 'average':
            # 배치 전체의 평균 클러스터 할당 반환
            prob_avg = prob.mean(dim=0)  # [n_vars, n_cluster]
            return prob_avg, cluster_centers, x_emb
    
class Hierarchical_cluster_assigner(nn.Module):
    def __init__(self, n_vars, n_cluster, seq_len, d_model, device='cuda', linkage='ward'):
        super(Hierarchical_cluster_assigner, self).__init__()
        self.n_vars = n_vars
        self.n_cluster = n_cluster
        self.seq_len = seq_len
        self.d_model = d_model
        self.device = device
        self.linkage = linkage

        # 각 채널을 임베딩 하기 위한 선형 레이어 선언
        self.linear = nn.Linear(seq_len, d_model)

        # Hierarchical 클러스터링을 위한 파라미터 (예: 덴드로그램의 높이 정보 등)
        # 실제 구현에 따라 필요한 파라미터를 추가
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=False)  # 클러스터링 임계값

    def hierarchical_clustering(self, x_emb):
        '''
        Torch 기반 Hierarchical 클러스터링
        x_emb: [N, d_model] - 임베딩된 채널들
        return: 
            - cluster_assignments [N, n_cluster] - 각 샘플의 클러스터 할당 (one-hot)
            - cluster_centers [n_cluster, d_model] - 각 클러스터의 중심
        '''
        N = x_emb.shape[0]
        
        # 모든 포인트 간의 거리 행렬 계산
        distances = torch.cdist(x_emb, x_emb, p=2)  # [N, N]
        
        # 초기화
        cluster_labels = torch.zeros(N, dtype=torch.long, device=x_emb.device)  # 클러스터 레이블 (0부터 시작)
        current_cluster = 0
        
        # 연결 기준에 따라 클러스터링 수행
        if self.linkage == 'ward':
            # Ward의 연결 기준: 클러스터 간 거리 최소화
            from scipy.cluster.hierarchy import linkage as sch_linkage, fcluster
            import numpy as np

            # SciPy의 계층적 군집화 사용
            linkage_matrix = sch_linkage(distances.cpu().numpy(), method='ward')
            
            # 덴드로그램 잘라내기 (클러스터 수에 따라)
            cluster_ids = fcluster(linkage_matrix, t=self.n_cluster, criterion='maxclust')
            
            # 클러스터 레이블 매핑
            cluster_labels = torch.tensor(cluster_ids, device=x_emb.device) - 1  # 0부터 시작하도록 조정

        elif self.linkage == 'average':
            # 평균 연결: 클러스터 간 평균 거리 최소화
            pass  # 구현 필요

        elif self.linkage == 'complete':
            # 완전 연결: 클러스터 간 최대 거리 최소화
            pass  # 구현 필요

        elif self.linkage == 'single':
            # 단일 연결: 클러스터 간 최소 거리 최소화
            pass  # 구현 필요

        # 클러스터 할당 결과를 reshape
        cluster_assignments = F.one_hot(cluster_labels, num_classes=self.n_cluster).float()
        
        # 각 클러스터의 중심 계산
        cluster_centers = torch.zeros(self.n_cluster, self.d_model, device=x_emb.device)
        for k in range(self.n_cluster):
            mask = (cluster_labels == k)
            if mask.sum() > 0:
                cluster_centers[k] = x_emb[mask].mean(dim=0)
            else:
                # 빈 클러스터는 랜덤 포인트로 초기화
                cluster_centers[k] = x_emb[torch.randint(0, N, (1,))].squeeze(0)
        
        return cluster_assignments, cluster_centers

    def forward(self, x, return_type='individual'):
        '''
        x: [bs, seq_len, n_vars]
        return_type: 'individual' or 'average'
        return: 
            - prob: [bs, n_vars, n_cluster] if individual, [n_vars, n_cluster] if average
            - cluster_centers: [n_cluster, d_model]
            - x_emb: [bs*n_vars, d_model]
        '''
        n_vars = x.shape[-1]
        x = x.permute(0, 2, 1)  # [bs, n_vars, seq_len]
        x_emb = self.linear(x).reshape(-1, self.d_model)  # [bs*n_vars, d_model]
        bn = x_emb.shape[0]
        bs = max(int(bn / n_vars), 1)

        # Hierarchical 클러스터링 수행
        cluster_assignments, cluster_centers = self.hierarchical_clustering(x_emb)
        
        # 클러스터 할당 결과를 reshape
        prob = cluster_assignments.reshape(bs, n_vars, self.n_cluster)  # [bs, n_vars, n_cluster]

        if return_type == 'individual':
            # 각 배치의 개별 클러스터 할당 반환
            return prob, cluster_centers, x_emb
        elif return_type == 'average':
            # 배치 전체의 평균 클러스터 할당 반환
            prob_avg = prob.mean(dim=0)  # [n_vars, n_cluster]
            return prob_avg, cluster_centers, x_emb