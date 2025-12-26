import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.data import Data, Batch


class FacialGNN(nn.Module):
    def __init__(self, hidden_dims=[32, 64, 128], fusion_dim=None, dropout=0.2):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.fusion_dim = fusion_dim 
        self.use_fusion = fusion_dim is not None
        
        self.gat1 = GATConv(in_channels=2, out_channels=hidden_dims[0] // 4, heads=4, concat=True, 
                            dropout=dropout, add_self_loops=True)
        self.gat2 = GATConv(in_channels=hidden_dims[0], out_channels=hidden_dims[1] // 4, heads=4, concat=True, 
                            dropout=dropout, add_self_loops=True)
        self.gat3 = GATConv(in_channels=hidden_dims[1], out_channels=hidden_dims[2], heads=1, concat=False, 
                            dropout=dropout, add_self_loops=True)
        
        self.norm1 = LayerNorm(hidden_dims[0])
        self.norm2 = LayerNorm(hidden_dims[1])
        self.norm3 = LayerNorm(hidden_dims[2])
        self.dropout = nn.Dropout(dropout)

        if self.use_fusion:
            self.proj1 = nn.Linear(hidden_dims[0], fusion_dim)
            self.proj2 = nn.Linear(hidden_dims[1], fusion_dim)
            self.proj3 = nn.Linear(hidden_dims[2], fusion_dim)
            
            self.alpha_mlp1 = nn.Linear(fusion_dim, 1)
            self.alpha_mlp2 = nn.Linear(fusion_dim, 1)
            self.alpha_mlp3 = nn.Linear(fusion_dim, 1)
        else:
            self.final_dim = hidden_dims[2]

    def forward(self, x, edge_index, batch):
        x1 = self.gat1(x, edge_index)
        x1 = self.norm1(x1, batch)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        if self.use_fusion:
            g1 = global_mean_pool(x1, batch)
            proj_g1 = self.proj1(g1)

        x2 = self.gat2(x1, edge_index)
        x2 = self.norm2(x2, batch)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        
        if self.use_fusion:
            g2 = global_mean_pool(x2, batch)
            proj_g2 = self.proj2(g2)

        x3 = self.gat3(x2, edge_index)
        x3 = self.norm3(x3, batch)
        x3 = F.relu(x3)
        if self.use_fusion:
            x3 = self.dropout(x3)
        
        if self.use_fusion:
            g3 = global_mean_pool(x3, batch)
            proj_g3 = self.proj3(g3)
            
            alpha1 = torch.sigmoid(self.alpha_mlp1(proj_g1))
            alpha2 = torch.sigmoid(self.alpha_mlp2(proj_g2))
            alpha3 = torch.sigmoid(self.alpha_mlp3(proj_g3))
            
            fused = alpha1 * proj_g1 + alpha2 * proj_g2 + alpha3 * proj_g3
            return fused
        else:
            return global_mean_pool(x3, batch)
        

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class FacialTemporalTransformer(nn.Module):
    def __init__(self, embed_dim: int = 128,
            num_layers: int = 4,
            num_heads: int = 8,
            ff_dim: int = None,
            dropout: float = 0.1,
            use_cls_token: bool = True
        ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim or 4 * embed_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        """
        x: [batch_size, seq_len, embed_dim]
        """
        batch_size = x.shape[0]
        
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, embed_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [batch, 1 + seq_len, embed_dim]
        
        x = self.pos_encoding(x)
        
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        if self.use_cls_token:
            representation = x[:, 0]
        else:
            representation = x.mean(dim=1)
        
        return representation
    

class SequencePadder(nn.Module):
    def forward(self, seqs, lengths):
        max_len = max(lengths)
        batch_size = len(seqs)
        device = seqs[0].device
        embed_dim = seqs[0].shape[-1]

        padded_seqs = torch.zeros(batch_size, max_len, embed_dim, device=device)
        padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)

        for i, (seq, length) in enumerate(zip(seqs, lengths)):
            padded_seqs[i, :length] = seq
            padding_mask[i, :length] = False

        return padded_seqs, padding_mask
    
class MicroExpressionModel(nn.Module):
    def __init__(
        self,
        gnn: nn.Module,
        transformer: nn.Module,
        num_classes: int,
        embed_dim: int = 128
    ):
        super().__init__()
        self.gnn = gnn
        self.transformer = transformer
        self.padder = SequencePadder()
        self.classifier = nn.Linear(embed_dim, num_classes)

    @staticmethod
    def build_graph(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        selected_points = points[17:68] 
        edge_list = []

        parts = {
            'left_brow': [17, 18, 19, 20, 21],
            'right_brow': [22, 23, 24, 25, 26],
            'nose_bridge': [27, 28, 29, 30],
            'nose_bottom': [31, 32, 33, 34, 35],
            'left_eye': [36, 37, 38, 39, 40, 41],
            'right_eye': [42, 43, 44, 45, 46, 47],
            'mouth_outer': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            'mouth_inner': [60, 61, 62, 63, 64, 65, 66, 67],
        }

        edge_list = []
        for part in parts.values():
            for i in range(len(part) - 1):
                edge_list.append((part[i], part[i + 1]))

        edge_list.extend([(36, 41), (42, 47), (48, 59), (60, 67)])
        edge_list.extend([(21, 27), (22, 27), (30, 33)])
        edge_list.extend([(17, 36), (21, 39), (22, 42), (26, 45)])
        edge_list.extend([(38, 27), (39, 27), (40, 27), (42, 27), (43, 27), (47, 27)])
        edge_list.extend([(36, 31), (39, 31), (40, 31), (41, 31)])
        edge_list.extend([(42, 35), (45, 35), (46, 35), (47, 35)])
        edge_list.extend([(48, 31), (49, 31), (53, 35), (54, 35)])
        edge_list.extend([(48, 60), (54, 64)])

        edge_list_reverse = [(end, start) for start, end in edge_list]
        edge_list.extend(edge_list_reverse)

        return selected_points, (torch.tensor(edge_list, device=points.device).T - 17)
    
    def forward(self, landmarks_seqs, lengths):
        device = next(self.parameters()).device
        batch_size = len(landmarks_seqs)

        all_landmarks = []
        cum_lengths = torch.cumsum(lengths, dim=0)
        start_idx = torch.cat([torch.tensor([0], device=device), cum_lengths[:-1]])
        
        for seq in landmarks_seqs:
            all_landmarks.extend(seq)

        data_list = []
        for points in all_landmarks:
            points = points.to(device)
            x, edge_index = MicroExpressionModel.build_graph(points)
            data = Data(x=x, edge_index=edge_index.to(device))
            data_list.append(data)

        big_batch = Batch.from_data_list(data_list).to(device)

        frame_embeds = self.gnn(big_batch.x, big_batch.edge_index, big_batch.batch)

        seq_embeds = []
        for i in range(batch_size):
            start = start_idx[i].item()
            end = cum_lengths[i].item()
            seq = frame_embeds[start:end]
            seq_embeds.append(seq)

        padded_seqs, padding_mask = self.padder(seq_embeds, lengths.to(device))

        if self.transformer.use_cls_token:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        representation = self.transformer(padded_seqs, padding_mask=padding_mask)

        logits = self.classifier(representation)

        return logits
    
