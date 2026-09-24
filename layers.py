import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x


class CrissCrossTransformerLayer(nn.Module):
    """Pre-norm transformer layer with criss-cross attention (CBraMod).

    Half of the feature dimension attends across channels (spatial) and the other half
    attends across patches (temporal, causal mask). Input shape: (B, C, P, D).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.gelu,
                 layer_norm_eps=1e-5, bias=True):
        super().__init__()
        self.transformer_channel = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout,
                                                         bias=bias, batch_first=True)
        self.transformer_temporal = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout,
                                                          bias=bias, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, padding_mask=None):
        x = x + self._criss_cross_block(self.norm1(x), padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _criss_cross_block(self, x, padding_mask=None):
        bz, ch_num, patch_num, dim = x.shape
        xs = x[:, :, :, :dim // 2]
        xt = x[:, :, :, dim // 2:]

        # spatial attention over channels
        xs = xs.transpose(1, 2).contiguous().view(bz * patch_num, ch_num, dim // 2)
        if padding_mask is not None:
            # NOTE: kept as a float mask (added to the attention logits) to match the pre-trained weights
            padding_mask = padding_mask.unsqueeze(1).expand(-1, patch_num, -1).reshape(bz * patch_num, ch_num).float()
        xs = self.transformer_channel(xs, xs, xs, need_weights=False, key_padding_mask=padding_mask)[0]
        xs = xs.contiguous().view(bz, patch_num, ch_num, dim // 2).transpose(1, 2)

        # temporal attention over patches
        xt = xt.contiguous().view(bz * ch_num, patch_num, dim // 2)
        attn_mask = torch.triu(torch.ones(patch_num, patch_num, device=x.device), diagonal=1).bool()
        xt = self.transformer_temporal(xt, xt, xt, need_weights=False, attn_mask=attn_mask)[0]
        xt = xt.contiguous().view(bz, ch_num, patch_num, dim // 2)

        return self.dropout1(torch.cat((xs, xt), dim=3))

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
