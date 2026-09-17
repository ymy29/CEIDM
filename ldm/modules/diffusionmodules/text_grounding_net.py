import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F

from ldm.modules.x_transformer import AbsolutePositionalEmbedding, FixedPositionalEmbedding

from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans


class HOIPositionNetV5(nn.Module):
    """
    Transform interaction information into interaction condition tokens
    """
    def __init__(self, in_dim, out_dim, fourier_freqs=8, max_interactions=30):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.interaction_embedding = AbsolutePositionalEmbedding(dim=out_dim, max_seq_len=max_interactions)
        self.position_embedding = AbsolutePositionalEmbedding(dim=out_dim, max_seq_len=3)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        self.linears = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.linear_action = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_action_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def get_between_box(self, bbox1, bbox2):
        """ Between Set Operation
        Operation of Box A between Box B from Prof. Jiang idea
        """
        all_x = torch.cat([bbox1[:, :, 0::2], bbox2[:, :, 0::2]],dim=-1)
        all_y = torch.cat([bbox1[:, :, 1::2], bbox2[:, :, 1::2]],dim=-1)
        all_x, _ = all_x.sort()
        all_y, _ = all_y.sort()
        return torch.stack([all_x[:,:,1], all_y[:,:,1], all_x[:,:,2], all_y[:,:,2]],2)

    # Interactive action clustering and offset information 
    # (1) The offset direction is generated using the clustering method, and the clustering operation is performed by the action feature.
    def generate_semantic_directions(self, action_positive_embeddings):
        """
        K-means clustering is used to generate semantic directions and retain local semantic similarity
        :param action_positive_embeddings: Action feature tensor (B,30,C)
        :return: Cluster center direction vector (B, K, C)
        """
        B, N, C = action_positive_embeddings.shape
        device = action_positive_embeddings.device
    
        features_flat = action_positive_embeddings.view(-1, C).cpu().numpy()

        K = min(2, len(np.unique(features_flat, axis=0)))  #Dynamically adjust the number of clusters

        try:
            kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)  # Explicitly set n_init
            kmeans.fit(features_flat)
            cluster_centers = torch.tensor(kmeans.cluster_centers_, device=device, dtype=torch.float32)
        except ValueError:
            cluster_centers = features_flat.mean(axis=0).reshape(1, -1)
            directions = np.zeros_like(features_flat)
            return (
                torch.tensor(cluster_centers, device=device), 
                torch.tensor(directions, device=device)
            )
        
        sample_to_center = action_positive_embeddings.unsqueeze(2) - cluster_centers.unsqueeze(0).unsqueeze(0)  # (B, N, K, C)，表示每个样本到每个聚类中心的向量。
        
        distances = torch.norm(sample_to_center, dim=-1)  # (B, N, K)
        nearest_cluster = distances.argmin(dim=-1, keepdim=True)  # (B, N, 1)
        
        directions = torch.gather(
            sample_to_center,  # (B, N, K, C)
            dim=2,
            index=nearest_cluster.unsqueeze(-1).expand(-1, -1, -1, C)  # (B, N, 1, C)
        ).squeeze(2)  # (B, N, C)
        
        directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
        
        return cluster_centers, directions
    
    def forward(self, subject_boxes, object_boxes, masks,
                subject_positive_embeddings, object_positive_embeddings, action_positive_embeddings):
        B, N, _ = subject_boxes.shape
        masks = masks.unsqueeze(-1)

        # embedding position (it may include padding as placeholder)
        action_boxes = self.get_between_box(subject_boxes, object_boxes)
        subject_xyxy_embedding = self.fourier_embedder(subject_boxes)  # B*N*4 --> B*N*C
        object_xyxy_embedding = self.fourier_embedder(object_boxes)  # B*N*4 --> B*N*C
        action_xyxy_embedding = self.fourier_embedder(action_boxes)  # B*N*4 --> B*N*C

        # learnable null embedding
        positive_null = self.null_positive_feature.view(1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)
        action_null = self.null_action_feature.view(1, 1, -1)

        # replace padding with learnable null embedding
        subject_positive_embeddings = subject_positive_embeddings * masks + (1 - masks) * positive_null
        object_positive_embeddings = object_positive_embeddings * masks + (1 - masks) * positive_null

        subject_xyxy_embedding = subject_xyxy_embedding * masks + (1 - masks) * xyxy_null
        object_xyxy_embedding = object_xyxy_embedding * masks + (1 - masks) * xyxy_null
        action_xyxy_embedding = action_xyxy_embedding * masks + (1 - masks) * xyxy_null

        action_positive_embeddings = action_positive_embeddings * masks + (1 - masks) * action_null

        objs_subject = self.linears(torch.cat([subject_positive_embeddings, subject_xyxy_embedding], dim=-1))
        objs_object = self.linears(torch.cat([object_positive_embeddings, object_xyxy_embedding], dim=-1))
        objs_action = self.linear_action(torch.cat([action_positive_embeddings, action_xyxy_embedding], dim=-1))
        #action_positive_embeddings = objs_action

        objs_subject = objs_subject + self.interaction_embedding(objs_subject)
        objs_object = objs_object + self.interaction_embedding(objs_object)
        objs_action = objs_action + self.interaction_embedding(objs_action) #instance embedding

        objs_subject = objs_subject + self.position_embedding.emb(torch.tensor(0).to(objs_subject.device))
        objs_object = objs_object + self.position_embedding.emb(torch.tensor(1).to(objs_object.device))
        objs_action = objs_action + self.position_embedding.emb(torch.tensor(2).to(objs_action.device))# role embedding

        objs1 = torch.cat([objs_subject, objs_action, objs_object], dim=1)

        print("action_positive_embeddings shape:", action_positive_embeddings.shape)

        # (2) Generate the offset action feature
        cluster_centers, sample_directions = self.generate_semantic_directions(action_positive_embeddings)  # [K, C], [B, N, C]
        offset_magnitudes = torch.tensor([-0.1, -0.05, 0.05, 0.1, 0.15, 0.2]).view(-1, 1, 1).to(cluster_centers.device)
        
        global_directions = cluster_centers.unsqueeze(0).expand(B, -1, -1)  # (B, K, C)

        action_embeddings_list = []
        for offset in offset_magnitudes:
            global_offsets = [
                action_positive_embeddings + offset * global_dir
                for global_dir in global_directions.unbind(dim=1)
            ]
            local_offsets = [
                action_positive_embeddings + offset * local_dir 
                for local_dir in sample_directions.unbind(dim=0) 
            ]
            
            action_embeddings_list.extend(global_offsets + local_offsets)

        # (3) Generate objs_action for each offset action feature
        objs_actions = []
        for idx, action_emb in enumerate(action_embeddings_list):

            objs_action1 = self.linear_action(
                torch.cat([action_emb, action_xyxy_embedding], dim=-1)
            )
            objs_action1 = objs_action1 + self.interaction_embedding(objs_action1)
            objs_action1 = objs_action1 + self.position_embedding.emb(
                torch.tensor(2).to(objs_action1.device)
            )
            objs_actions.append(objs_action1)

        # (4) After objs_actions is generated, use the list derivation to create the objs after each offset
        objs_list = [
            torch.cat([objs_subject, act, objs_object], dim=1) 
            for act in objs_actions
        ]

        objs = torch.cat([objs1] + objs_list, dim=1) 
        print(f"objs shape: {objs.shape}") 
        return objs, objs1, subject_boxes, object_boxes, action_boxes, subject_positive_embeddings, object_positive_embeddings
 # objs1 is explicit interactive information; objs is action offset interactive information.