import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.spatial_proj = nn.Sequential(
            nn.Conv1d(channels, max(channels // 4, 2), kernel_size=1),  # Modified divider for fewer channels
            nn.BatchNorm1d(max(channels // 4, 2)),
            nn.ReLU(),
            nn.Conv1d(max(channels // 4, 2), channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Weight each channel
        attn = self.spatial_proj(x)
        return x * attn

class MSSTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.5, initialization='xavier'):
        super(MSSTBlock, self).__init__()
        
        # Spatial filtering (channel attention)
        self.spatial_filter = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            SpatialAttention(out_channels)
        )
        
        # Temporal filtering with specific kernel size
        self.temporal_filter = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # Initialize weights properly
        self._init_weights(initialization)
    
    def _init_weights(self, initialization):
        if initialization == 'xavier':
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, channels, time_points)
        x = self.spatial_filter(x)
        x = self.temporal_filter(x)
        return x

class ScaleAttention(nn.Module):
    def __init__(self, channels):
        super(ScaleAttention, self).__init__()
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, max(channels // 4, 2), kernel_size=1),  # Modified divider for fewer channels
            nn.ReLU(),
            nn.Conv1d(max(channels // 4, 2), channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attn = self.scale_attn(x)
        return x * attn

class CSGTEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=2, num_heads=2, dropout=0.5, use_layer_norm=True):
        super(CSGTEncoder, self).__init__()
        
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        
        # Initial projection
        self.input_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        
        # Stack of transformer layers
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,  # Reduced feedforward dimension
                dropout=dropout,
                activation="gelu",
                batch_first=True
            )
            self.transformer_layers.append(layer)
        
        # Layer normalization if required
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x shape: (batch_size, channels, time_points)
        
        # Project input
        x = self.input_proj(x)  # (batch_size, hidden_dim, time_points)
        
        # Prepare for transformer: (batch_size, seq_len, hidden_dim)
        x = x.permute(0, 2, 1)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Apply layer normalization if specified
        if self.use_layer_norm:
            x = self.layer_norm(x)
        
        # Return to original shape: (batch_size, hidden_dim, time_points)
        x = x.permute(0, 2, 1)
        
        return x

class MSVTNet_BCIC_IV_2b(nn.Module):
    def __init__(self, num_channels=6, num_classes=2, input_time_length=876, dropout_rate=0.5):
        super(MSVTNet_BCIC_IV_2b, self).__init__()
        
        # Tracking training step for dynamic loss weighting
        self.training_step = 0
        
        # Architecture parameters - adjusted for fewer channels
        self.embedding_dim = 16  # Reduced from 32 for smaller input
        
        # MSST Blocks with different kernel sizes for multi-scale features
        self.msst_blocks = nn.ModuleList([
            MSSTBlock(
                in_channels=num_channels, 
                out_channels=self.embedding_dim,
                kernel_size=k,
                dropout=dropout_rate,
                initialization='xavier'
            ) for k in [3, 5, 7]  # Multiple kernel sizes for different scales
        ])
        
        # Scale attention to weight features from different scales
        self.scale_attention = ScaleAttention(self.embedding_dim * 3)
        
        # CSGT Encoder for feature integration - with reduced complexity
        self.csgt_encoder = CSGTEncoder(
            in_dim=self.embedding_dim * 3,  # Combined from MSST blocks
            hidden_dim=32,  # Reduced from 64
            num_layers=2,
            num_heads=2,  # Reduced from 4 since we have fewer channels
            dropout=dropout_rate,
            use_layer_norm=True
        )
        
        # Classification head - adjusted for binary classification
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten after adaptive pooling
            nn.Linear(32, 16),  # Reduced dimensions
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, num_classes)
        )
        
        # Auxiliary classifiers for each MSST scale
        self.aux_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(self.embedding_dim, num_classes)
            ) for _ in range(len(self.msst_blocks))
        ])
    
    def forward(self, x):
        # Process through MSST blocks - multi-scale feature extraction
        msst_outputs = []
        for msst_block in self.msst_blocks:
            msst_outputs.append(msst_block(x))
        
        # Calculate auxiliary outputs before fusion
        aux_outputs = []
        for i, output in enumerate(msst_outputs):
            # Global average pooling + classification
            aux_outputs.append(self.aux_classifiers[i](output))
        
        # Feature fusion - concatenate along channel dimension
        merged_features = torch.cat(msst_outputs, dim=1)  # (batch_size, channels*3, time)
        
        # Apply attention across scales before CSGT
        merged_features = self.scale_attention(merged_features)
        
        # CSGT encoding
        encoded_features = self.csgt_encoder(merged_features)
        
        # Global average pooling
        gap_features = F.adaptive_avg_pool1d(encoded_features, 1)
        
        # Final classification
        logits = self.classifier(gap_features)
        
        return logits, aux_outputs
    
    def extract_features(self, x):
        """Extract features before final classification for visualization"""
        # Process through MSST blocks
        msst_outputs = []
        for msst_block in self.msst_blocks:
            msst_outputs.append(msst_block(x))
        
        # Feature fusion
        merged_features = torch.cat(msst_outputs, dim=1)
        merged_features = self.scale_attention(merged_features)
        
        # CSGT encoding
        encoded_features = self.csgt_encoder(merged_features)
        
        # Global average pooling
        features = F.adaptive_avg_pool1d(encoded_features, 1).squeeze(-1)
        
        return features
    
    def compute_loss(self, outputs, aux_outputs, targets, weights=None):
        # Dynamic loss weighting based on training progress
        if self.training_step < 1000:
            alpha = 0.7  # Slightly reduced compared to original due to binary task
        else:
            alpha = 0.2  # Reduced compared to original
        
        # Main classification loss with optional class weighting
        if weights is not None:
            cls_loss = F.cross_entropy(outputs, targets, weight=weights)
        else:
            cls_loss = F.cross_entropy(outputs, targets)
        
        # Auxiliary losses with equal weighting
        aux_loss = 0
        for i, aux_output in enumerate(aux_outputs):
            if weights is not None:
                aux_loss += F.cross_entropy(aux_output, targets, weight=weights)
            else:
                aux_loss += F.cross_entropy(aux_output, targets)
        aux_loss /= len(aux_outputs) if aux_outputs else 1
        
        # Total loss with dynamic weighting
        total_loss = (1 - alpha) * cls_loss + alpha * aux_loss
        
        # Increment training step
        self.training_step += 1
        
        return total_loss, cls_loss, aux_loss, 0