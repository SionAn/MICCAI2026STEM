import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import CrissCrossTransformerLayer, TransformerEncoder


class STEM(nn.Module):
    """Subject- and Task-Aware EEG Foundation Model.

    Shared encoder -> (1) D_self for self-supervised contrastive learning,
                      (2) decomposer -> D_task / D_subject for meta-learning.

    Args:
        patch_size: length L of each temporal patch (40 samples = 0.2 s at 200 Hz).
        out_dim: output dimension of D_self projection.
        d_model: feature dimension D.
        n_layer: total number of transformer layers (encoder = n_layer - 2, D_self = 2).
        num_cls: number of downstream classes. If None, the model runs in pre-training mode.
        num_ch: number of downstream channels (needed for the classifier only).
    """

    def __init__(self, patch_size=40, out_dim=40, d_model=256, dim_feedforward=256, n_layer=6, nhead=8,
                 num_cls=None, num_ch=None):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size, d_model)
        layer = CrissCrossTransformerLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                           activation=F.gelu)
        self.encoder = TransformerEncoder(layer, num_layers=n_layer - 2)
        self.decoder = TransformerEncoder(layer, num_layers=2)  # D_self
        self.decomposer = Decomposer(d_model)

        half_layer = CrissCrossTransformerLayer(d_model=d_model // 2, nhead=nhead // 2,
                                                dim_feedforward=dim_feedforward, activation=F.gelu)
        self.encoder_task = TransformerEncoder(half_layer, num_layers=2)  # D_task
        self.encoder_sub = TransformerEncoder(half_layer, num_layers=2)  # D_subject
        self.proj_task = nn.Sequential(nn.Linear(d_model // 2, d_model // 2))
        self.proj_subject = nn.Sequential(nn.Linear(d_model // 2, d_model // 2))
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))

        self.classifier = None
        if num_cls:
            self.classifier = nn.Sequential(
                nn.Linear(num_ch * (d_model // 2), d_model // 2),
                nn.ELU(),
                nn.Linear(d_model // 2, d_model // 8),
                nn.ELU(),
                nn.Linear(d_model // 8, num_cls),
            )
        self.apply(_weights_init)

    def forward(self, x, ch, mask=None, padding_mask=None):
        """
        x: (B, C, P, L) EEG patches, ch: (B, C, 3) electrode coordinates,
        mask: (B, C, P) patch mask for SSL, padding_mask: (B, C) with 1 for padded channels.

        Pre-training mode returns (z_self, z_task, z_subject) with z_task / z_subject averaged over channels.
        Fine-tuning mode returns (logits, z_task, z_subject) with channel-wise features flattened.
        """
        z = self.patch_embedding(x, ch, mask)
        z = self.encoder(z, padding_mask)
        task, subject = self.decomposer(z.permute(0, 3, 2, 1))  # (B, D/2, P, C)

        z_task = self.proj_task(self.encoder_task(task.permute(0, 3, 2, 1), padding_mask)).mean(dim=2)
        z_sub = self.proj_subject(self.encoder_sub(subject.permute(0, 3, 2, 1), padding_mask)).mean(dim=2)

        if self.classifier is None:
            z_self = self.decoder(torch.cat([task, subject], dim=1).permute(0, 3, 2, 1), padding_mask)
            z_self = self.proj_out(z_self).mean(dim=2)
            return z_self.flatten(1), z_task.mean(dim=1), z_sub.mean(dim=1)

        out = self.classifier(z_task.flatten(1))
        return out, z_task.flatten(1), z_sub.flatten(1)


class Decomposer(nn.Module):
    """Splits encoder features into task-aware and subject-aware halves."""

    def __init__(self, nfeat):
        super().__init__()
        self.nfeat = nfeat
        self.embed_layer = nn.Sequential(nn.Conv2d(nfeat, nfeat, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(nfeat), nn.ELU(), nn.Dropout())

    def forward(self, x):
        task, subject = torch.split(self.embed_layer(x), [self.nfeat // 2, self.nfeat // 2], dim=1)
        return task, subject


class PatchEmbedding(nn.Module):
    """E = phi(x_p) + E_freq + E_seq + E_spa."""

    def __init__(self, patch_size, d_model):
        super().__init__()
        self.positional_encoding = nn.Sequential(nn.Linear(d_model, d_model), nn.ELU())
        self.temporal_encoding = nn.Sequential(nn.Linear(patch_size, d_model), nn.ELU())  # W_seq
        self.channel_encoding = nn.Sequential(nn.Linear(3, d_model), nn.ELU())  # W_spa
        self.mask_encoding = nn.Parameter(torch.zeros(patch_size), requires_grad=False)

        # phi: 3-layer CNN with kernel size 3 along the patch axis
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=patch_size, out_channels=d_model, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(64, d_model),
            nn.GELU(),

            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(64, d_model),
            nn.GELU(),

            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(64, d_model),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(  # W_freq
            nn.Linear(patch_size // 2 + 1, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x, ch, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        mask_x = x.float() if mask is None else x.clone().float()
        if mask is not None:
            mask_x[mask == 1] = self.mask_encoding

        patch_emb = self.proj_in(mask_x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()  # (B, C, P, D)

        spectral = torch.fft.rfft(mask_x.reshape(bz * ch_num * patch_num, patch_size), dim=-1, norm='forward')
        spectral = torch.abs(spectral).view(bz, ch_num, patch_num, -1)
        patch_emb = patch_emb + self.spectral_proj(spectral)

        temporal_embedding = self.temporal_encoding(sinusoidal_embedding(patch_num, patch_size, x.device))  # (1, P, D)
        channel_embedding = self.channel_encoding(ch)  # (B, C, D)
        positional_embedding = patch_emb + channel_embedding.unsqueeze(2) + temporal_embedding.unsqueeze(1)
        return patch_emb + self.positional_encoding(positional_embedding)


def sinusoidal_embedding(length, dim, device=None):
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros((length, dim), device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
