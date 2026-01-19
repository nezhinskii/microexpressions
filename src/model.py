import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.data import Data, Batch

class FacialGNN(nn.Module):
    def __init__(self, hidden_dims=[32, 64, 128], fusion_dim=None, dropout=0.2, pool='mean'):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.fusion_dim = fusion_dim 
        self.use_fusion = fusion_dim is not None
        self.pool = pool
        
        self.gats = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        in_channels = 5
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                out_channels = hidden_dims[i] // 4
                heads = 4
                concat = True
            else:
                out_channels = hidden_dims[i]
                heads = 1
                concat = False
            
            gat = GATConv(in_channels=in_channels, out_channels=out_channels, heads=heads, concat=concat, 
                          dropout=dropout, add_self_loops=True, edge_dim=6)
            self.gats.append(gat)
            
            norm = LayerNorm(hidden_dims[i])
            self.norms.append(norm)
            
            in_channels = hidden_dims[i]
            
        self.dropout = nn.Dropout(dropout)

        if self.use_fusion:
            self.projs = nn.ModuleList([nn.Linear(hdim, fusion_dim) for hdim in hidden_dims])
            self.alpha_mlps = nn.ModuleList([nn.Linear(fusion_dim, 1) for _ in hidden_dims])
        else:
            self.final_dim = hidden_dims[-1]

    def forward(self, x, edge_index, batch, edge_attr=None):
        gs = []
        if self.pool == 'mean':
            global_pool_op = global_mean_pool
        else:
            global_pool_op = global_max_pool
            
        for i in range(self.num_layers):
            x = self.gats[i](x, edge_index, edge_attr=edge_attr)
            x = self.norms[i](x, batch)
            x = F.relu(x)
            x = self.dropout(x)
            
            if self.use_fusion:
                g = global_pool_op(x, batch)
                proj_g = self.projs[i](g)
                gs.append(proj_g)
        
        if self.use_fusion:
            alphas = [torch.sigmoid(self.alpha_mlps[i](gs[i])) for i in range(self.num_layers)]
            fused = sum(alphas[i] * gs[i] for i in range(self.num_layers))
            return fused
        else:
            return global_pool_op(x, batch)

class MILPooling(nn.Module):
    def __init__(
        self,
        frame_dim: int = 256,
        d_model: int = 512,
        kernel_size: int = 3,
        num_conv_layers: int = 1,
        dropout: float = 0.3,
        temperature: float = 1.0,
        use_residual: bool = False,
    ):
        super().__init__()
        
        self.proj = nn.Linear(frame_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
        conv_layers = []
        for _ in range(num_conv_layers):
            conv_layers.append(
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2)
            )
        self.conv_stack = nn.Sequential(*conv_layers) if num_conv_layers > 0 else nn.Identity()
        
        self.norm = nn.LayerNorm(d_model)
        self.use_residual = use_residual and num_conv_layers > 0
        
        self.frame_scorer = nn.Linear(d_model, 1)
        self.temperature = temperature

    def forward(self, x, padding_mask=None, return_extras=False):
        """
        x: [batch_size, seq_len, embed_dim]
        """
        x = self.proj(x)
        
        residual = x if self.use_residual else None
        x = x.transpose(1, 2)
        x = self.conv_stack(x)
        x = x.transpose(1, 2)
        
        if self.use_residual:
            x = x + residual
            
        x = self.norm(x)
        
        # Attention
        scores = self.frame_scorer(x).squeeze(-1)
        scores = scores.masked_fill(padding_mask, float('-inf'))
        attn_weights = torch.softmax(scores / self.temperature, dim=1)
        attn_weights = attn_weights.masked_fill(padding_mask, 0.0)
        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        video_emb = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        video_emb = self.dropout(video_emb)
        
        if return_extras:
            return video_emb, scores, attn_weights
        else:
            return video_emb

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
    def __init__(self, 
            input_dim: int,
            embed_dim: int = 128,
            num_layers: int = 4,
            num_heads: int = 8,
            ff_dim: int = None,
            dropout: float = 0.1,
            use_cls_token: bool = True
        ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        
        self.input_proj = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        
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
        x = self.input_proj(x)          # [B, L, input_dim] → [B, L, embed_dim]
        
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
    
class FacialRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        rnn_type: str = 'gru',
    ):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.lower()
        
        if self.rnn_type == 'gru':
            rnn_class = nn.GRU
        elif self.rnn_type == 'lstm':
            rnn_class = nn.LSTM
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.output_dim = hidden_size * self.num_directions     
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Padded tensor of shape [batch_size, max_seq_len, input_size]
            lengths: Tensor of shape [batch_size] containing real sequence lengths
            
        Returns:
            representation: Tensor of shape [batch_size, output_dim]
                           Last hidden state of the RNN (concatenated if bidirectional)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        # For LSTM: hidden is a tuple (h_n, c_n) → take only h_n
        if self.rnn_type == 'lstm':
            hidden = hidden[0]
        # hidden shape: [num_layers * num_directions, batch_size, hidden_size]
        representation = hidden[-self.num_directions:, :, :]   # [num_directions, B, H]
        # Reshape: [B, num_directions, H] → [B, num_directions * H]
        representation = representation.permute(1, 0, 2).reshape(x.size(0), -1)
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
        temporal_module: nn.Module,
        num_classes: int,
        embed_dim: int = 128,
        temporal_model: str = 'transformer'
    ):
        super().__init__()
        self.gnn = gnn
        self.temporal_module = temporal_module
        self.padder = SequencePadder()
        self.temporal_model = temporal_model
        self.classifier = nn.Linear(embed_dim, num_classes)

    @staticmethod
    def build_graph(points: torch.Tensor, prev_points: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        def compute_anchor(pts: torch.Tensor) -> torch.Tensor:
            eye_left_inner = pts[39]
            eye_right_inner = pts[42]
            eye_center = (eye_left_inner + eye_right_inner) / 2
            nose_bridge = torch.mean(pts[27:31], dim=0)
            return (eye_center + nose_bridge) / 2
        selected_points = points[17:68]
        anchor = compute_anchor(points)
        delta_to_anchor = selected_points - anchor
        if prev_points is None:
            delta_prev = torch.zeros_like(selected_points)
            mag = torch.zeros(selected_points.shape[0], 1, device=points.device)
        else:
            prev_selected = prev_points[17:68]
            prev_anchor = compute_anchor(prev_points)
            prev_delta_to_anchor = prev_selected - prev_anchor
            delta_prev = delta_to_anchor - prev_delta_to_anchor
            mag = torch.norm(delta_prev, dim=1, keepdim=True)
            
        iod = torch.norm(points[39] - points[42]) + 1e-8
        mag = mag / iod
        
        verticies = torch.cat([
            delta_to_anchor,
            delta_prev,
            mag
        ], dim=1)
        
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
        edge_list.extend([(39, 27), (42, 27)])
        edge_list.extend([(36, 31), (39, 31)])
        edge_list.extend([(42, 35), (45, 35)])
        edge_list.extend([(48, 31), (54, 35)])
        edge_list.extend([(48, 60), (54, 64)])
        edge_list_reverse = [(end, start) for start, end in edge_list]
        edge_list.extend(edge_list_reverse)
        edge_index = torch.tensor(edge_list, device=points.device).T - 17
        
        def compute_normalized_lengths(selected_pts: torch.Tensor, full_pts: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            iod = torch.norm(full_pts[39] - full_pts[42]) + 1e-8
            src, dst = edge_index[0], edge_index[1]
            diffs = selected_pts[src] - selected_pts[dst]
            lengths = torch.norm(diffs, dim=1)
            angles = torch.atan2(diffs[:, 1], diffs[:, 0])
            sin_angles = torch.sin(2 * angles)
            cos_angles = torch.cos(2 * angles)
            return lengths / iod, sin_angles, cos_angles

        norm_lengths, sin_angles, cos_angles = compute_normalized_lengths(selected_points, points, edge_index)
        
        if prev_points is not None:
            prev_selected = prev_points[17:68]
            prev_norm_lengths, prev_sin, prev_cos = compute_normalized_lengths(prev_selected, prev_points, edge_index)
            
            delta_norm_lengths = norm_lengths - prev_norm_lengths
            
            delta_sin = sin_angles - prev_sin
            delta_cos = cos_angles - prev_cos
        else:
            delta_norm_lengths = torch.zeros_like(norm_lengths)
            delta_sin = torch.zeros_like(sin_angles)
            delta_cos = torch.zeros_like(cos_angles)

        edge_attr = torch.stack([norm_lengths, delta_norm_lengths, sin_angles, cos_angles, delta_sin, delta_cos], dim=1)
        
        return verticies, edge_index, edge_attr
    
    def forward(self, landmarks_seqs, lengths, return_extras=False):
        device = next(self.parameters()).device
        batch_size = len(landmarks_seqs)

        all_landmarks = []
        cum_lengths = torch.cumsum(lengths, dim=0)
        start_idx = torch.cat([torch.tensor([0], device=device), cum_lengths[:-1]])
        start_idx_set = set(start_idx.cpu().numpy().tolist())
        
        for seq in landmarks_seqs:
            all_landmarks.extend(seq)

        data_list = []
        for i, points in enumerate(all_landmarks):
            points = points.to(device)
            prev_points = None if i in start_idx_set else all_landmarks[i - 1].to(device)
            verticies, edge_index, edge_attr = MicroExpressionModel.build_graph(points, prev_points)
            data = Data(x=verticies, edge_index=edge_index.to(device), edge_attr=edge_attr.to(device))
            data_list.append(data)

        big_batch = Batch.from_data_list(data_list).to(device)

        frame_embeds = self.gnn(big_batch.x, big_batch.edge_index, big_batch.batch, big_batch.edge_attr)

        seq_embeds = []
        for i in range(batch_size):
            start = start_idx[i].item()
            end = cum_lengths[i].item()
            seq = frame_embeds[start:end]
            seq_embeds.append(seq)

        padded_seqs, padding_mask = self.padder(seq_embeds, lengths.to(device))

        if self.temporal_model == 'transformer':
            if self.temporal_module.use_cls_token:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=padding_mask.device)
                padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
            representation = self.temporal_module(padded_seqs, padding_mask=padding_mask)
            logits = self.classifier(representation)
            return logits
        
        elif self.temporal_model == 'mil':
            temporal_out = self.temporal_module(padded_seqs, padding_mask=padding_mask, return_extras=return_extras)
            if return_extras:
                representation, scores, attn_weights = temporal_out
            else:
                representation = temporal_out
            logits = self.classifier(representation)
            
            if return_extras:
                return logits, scores, attn_weights
            else:
                return logits
        elif self.temporal_model in ['gru', 'lstm']:
            representation = self.temporal_module(padded_seqs, lengths)
            logits = self.classifier(representation)
            return logits
        else:
            raise ValueError(f"Unknown temporal_model: {self.temporal_model}")

        
    
