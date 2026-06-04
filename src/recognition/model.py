import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SignRecognitionTransformer(nn.Module):
    def __init__(self, num_keypoints=27, d_model=256, nhead=8,
                 num_encoder_layers=6, vocab_size=2000, dropout=0.1):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        
        self.input_proj = nn.Linear(num_keypoints * 3, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # SPOTER-style learnable class query token
        self.class_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.decoder_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, tgt=None):
        # tgt is accepted for backward compatibility, but not used in encoder-only architecture
        batch_size = x.size(0)
        if x.dim() == 4:
            x = x.view(batch_size, x.size(1), -1)
            
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # Encoder pass
        encoder_out = self.transformer_encoder(x)
        
        # Expand class query to batch dimension
        query = self.class_query.expand(batch_size, -1, -1)
        
        # Decode Class Query over Encoder output
        attn_out, _ = self.decoder_attention(
            query=query,
            key=encoder_out,
            value=encoder_out
        )
        
        # Residual and LayerNorm
        out = self.layer_norm(query + self.dropout(attn_out))
        
        # Project to vocab
        out = self.output_proj(out.squeeze(1))
        return out
