import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super(MultiHeadAttention, self).__init__()
        self.nhead = cfg['nhead']
        self.attention = nn.MultiheadAttention(cfg['d_model'], self.nhead)

    def forward(self, q, k, v, attn_mask=None):
        return self.attention(q, k, v, key_padding_mask=attn_mask)[0]


class PositionWiseFeedForward(nn.Module):
    def __init__(self, cfg):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(cfg['d_model'], cfg['d_ff'])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(cfg['d_ff'], cfg['d_model'])

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(cfg)
        self.position_wise_feed_forward = PositionWiseFeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg['d_model'])
        self.norm2 = nn.LayerNorm(cfg['d_model'])
        self.dropout = nn.Dropout(cfg['dropout_rate'])

    def forward(self, x, attn_mask):
        attn_output = self.multi_head_attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.position_wise_feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DNATransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(DNATransformerEncoder, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        
        self.embed = nn.Embedding(6, self.d_model)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg['num_layers'])])
        self.linear = nn.Linear(self.d_model, 1)
        
    def forward(self, x, len_x, attn_mask=None):
        self.batch_size, self.seq_len = x.shape
        x = self.embed(x)
        attn_mask = self.create_mask(len_x).movedim(1,0) 
        x = x + self.get_position_encoding(self.seq_len, self.d_model)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = [torch.mean(x[i, :len_x[i], :], dim=0) for i in range(self.batch_size)]
        x = torch.stack(x)
        x = self.linear(x)
        return x
    
    def create_mask(self, len_x):
        mask_pad = torch.zeros(self.batch_size, self.seq_len).bool()
        for (idx, len_comment) in enumerate(len_x): 
            mask_pad[idx, len_comment:self.seq_len] = 1
        return mask_pad.to(self.device)
    
    def get_position_encoding(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i//2)) / d_model)
        pos_enc = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                theta = get_angle(pos, i, d_model)
                if i % 2 == 0:
                    pos_enc[pos, i] = np.sin(theta)
                else:
                    pos_enc[pos, i] = np.cos(theta)
        return torch.FloatTensor(pos_enc).to(self.device)
