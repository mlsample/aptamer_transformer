import torch
import torch.nn as nn
import numpy as np
import os
from x_transformers import XTransformer, TransformerWrapper, Decoder, Encoder

os.environ["TOKENIZERS_PARALLELISM"] = 'true'
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from mlguess.torch.class_losses import relu_evidence
    

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super(MultiHeadAttention, self).__init__()
        self.nhead = cfg['nhead']
        self.attention = nn.MultiheadAttention(cfg['d_model'], self.nhead, batch_first=True)

    def forward(self, q, k, v, attn_mask=None):
        return self.attention(q, k, v, key_padding_mask=attn_mask)[0]


class PositionWiseFeedForward(nn.Module):
    def __init__(self, cfg):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(cfg['d_model'], cfg['d_ff'], bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(cfg['d_ff'], cfg['d_model'], bias=False)

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
    
class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(cfg)
        self.enc_dec_attention = MultiHeadAttention(cfg)
        self.position_wise_feed_forward = PositionWiseFeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg['d_model'])
        self.norm2 = nn.LayerNorm(cfg['d_model'])
        self.norm3 = nn.LayerNorm(cfg['d_model'])
        self.dropout = nn.Dropout(cfg['dropout_rate'])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Masked multi-head attention
        attn_output = self.multi_head_attention(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Encoder-Decoder attention
        attn_output = self.enc_dec_attention(x, enc_output, enc_output, attn_mask=src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed forward
        ff_output = self.position_wise_feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
def get_position_encoding(seq_len, d_model):
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
    return torch.FloatTensor(pos_enc)

class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_path'])
        cfg['num_tokens'] = self.tokenizer.vocab_size
        
        self.embed = nn.Embedding(cfg['num_tokens'], self.d_model, padding_idx=0)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg['num_layers'])])
        
    def forward(self, x, attn_mask=None):
        self.batch_size, self.seq_len = x.shape
        
        x = self.embed(x)
        x = x + get_position_encoding(self.seq_len, self.d_model).to(self.device)
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super(TransformerDecoder, self).__init__()
        self.cfg = cfg
        self.d_model = cfg['d_model']
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_path'])
        cfg['num_tokens'] = self.tokenizer.vocab_size
        
        self.embed = nn.Embedding(cfg['num_tokens'], self.d_model, padding_idx=0)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg['num_layers'])])
        self.linear = nn.Linear(self.d_model, cfg['num_tokens'])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        self.batch_size, self.seq_len = x.shape
        
        x = self.embed(x)
        x = x + get_position_encoding(self.seq_len, self.d_model).to(self.cfg['device'])
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        x = self.linear(x)
        return x


class DNATransformerEncoderClassifier(nn.Module):
    def __init__(self, cfg):
        super(DNATransformerEncoderClassifier, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        
        self.transformer_encoder = TransformerEncoder(cfg)
        self.linear = nn.Linear(self.d_model, cfg['num_classes'])
        
    def forward(self, x, attn_mask=None):
        x = self.transformer_encoder(x, attn_mask=attn_mask)
        
        x = x[:, 0, :]
        
        x = self.linear(x)
        
        return x
    
class DNATransformerEncoderEvidence(nn.Module):
    def __init__(self, cfg):
        super(DNATransformerEncoderEvidence, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        self.n_classes = cfg['num_classes']
        
        self.transformer_encoder = TransformerEncoder(cfg)
        self.linear = nn.Linear(self.d_model, cfg['num_classes'])
        
    def forward(self, x, attn_mask=None):
        x = self.transformer_encoder(x, attn_mask=attn_mask)
        
        x = x[:, 0, :]
        
        x = self.linear(x)
        
        return x
    
    def predict_uncertainty(self, input_ids, attn_mask=None):
        y_pred = self(input_ids, attn_mask=attn_mask)
        
        # dempster-shafer theory
        evidence = relu_evidence(y_pred) # can also try softplus and exp evidence schemes
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        u = self.n_classes / S
        prob = alpha / S
        
        # law of total uncertainty 
        epistemic = prob * (1 - prob) / (S + 1)
        aleatoric = prob - prob**2 - epistemic
        return prob, u, aleatoric, epistemic
    
class DNATransformerEncoderRegression(nn.Module):
    def __init__(self, cfg):
        super(DNATransformerEncoderRegression, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        
        self.transformer_encoder = TransformerEncoder(cfg)
        self.linear = nn.Linear(self.d_model, 1)
        
    def forward(self, x, attn_mask=None):
        
        x = self.transformer_encoder(x, attn_mask=attn_mask)

        x = x[:, 0, :]
        
        x = self.linear(x)
        
        return x
    
class AptamerBert(nn.Module):
    def __init__(self, cfg):
        super(AptamerBert, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        
        os.environ["TOKENIZERS_PARALLELISM"] = 'true'

        self.tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_path'])
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,  mlm_probability=cfg['mlm_probability'], return_tensors='pt')
        
        self.transformer_encoder = TransformerEncoder(cfg)
        self.linear = nn.Linear(self.d_model, cfg['num_tokens'])
        
    def forward(self, x, attn_mask=None):

        embed = x = self.transformer_encoder(x, attn_mask=attn_mask)
        
        logits = self.linear(x)
        
        return logits, embed
    

class AptamerBertClassifier(nn.Module):
    def __init__(self, cfg):
        super(AptamerBertClassifier, self).__init__()

        self.aptamer_bert_encoding = AptamerBert(cfg)
        self.aptamer_bert_encoding.load_state_dict(
            torch.load(cfg['aptamer_bert_path'], map_location=cfg['device']
                )['model_state_dict']
            )
        
        self.linear = nn.Linear(cfg['d_model'], cfg['num_classes'])
        
    def forward(self, x, attn_mask=None):
        logits, embed = self.aptamer_bert_encoding(x, attn_mask=attn_mask)
        
        x = embed[:, 0, :]
        
        return self.linear(x)

class AptamerBertEvidence(nn.Module):
    def __init__(self, cfg):
        super(AptamerBertEvidence, self).__init__()

        self.aptamer_bert_encoding = AptamerBert(cfg)
        self.aptamer_bert_encoding.load_state_dict(
            torch.load(cfg['aptamer_bert_path'], map_location=cfg['device']
                )['model_state_dict']
            )
        
        self.linear = nn.Linear(cfg['d_model'], cfg['num_classes'])
        self.n_classes = cfg['num_classes']
    def forward(self, x, attn_mask=None):
        logits, embed = self.aptamer_bert_encoding(x, attn_mask=attn_mask)
        
        x = embed[:, 0, :]
        
        return self.linear(x)
    
    def predict_uncertainty(self, input_ids, attn_mask=None):
        y_pred = self(input_ids, attn_mask=attn_mask)
        
        # dempster-shafer theory
        evidence = relu_evidence(y_pred) # can also try softplus and exp evidence schemes
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        u = self.n_classes / S
        prob = alpha / S
        
        # law of total uncertainty 
        epistemic = prob * (1 - prob) / (S + 1)
        aleatoric = prob - prob**2 - epistemic
        return prob, u, aleatoric, epistemic

class XTransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(XTransformerEncoder, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        
        self.x_transformer_encoder= TransformerWrapper(
            num_tokens = cfg['num_tokens'],
            max_seq_len = cfg['max_seq_len'],
            num_memory_tokens = cfg['num_memory_tokens'],
            l2norm_embed = cfg['l2norm_embed'],
            attn_layers = Encoder(
                dim = cfg['d_model'],
                depth = cfg['num_layers'],
                heads = cfg['nhead'],
                layer_dropout = cfg['dropout_rate'],   # stochastic depth - dropout entire layer
                attn_dropout = cfg['dropout_rate'],    # dropout post-attention
                ff_dropout = cfg['dropout_rate'],       # feedforward dropout,
                attn_flash = cfg['attn_flash'],
                attn_num_mem_kv = cfg['attn_num_mem_kv'],
                use_scalenorm = cfg['use_scalenorm'],
                use_simple_rmsnorm = cfg['use_simple_rmsnorm'],
                ff_glu = cfg['ff_glu'],
                ff_swish = cfg['ff_swish'],
                ff_no_bias = cfg['ff_no_bias'],
                attn_talking_heads = cfg['attn_talking_heads']
            )
        )

    def forward(self, x, attn_mask=None):
        logits, embed = self.x_transformer_encoder(x, mask=attn_mask, return_logits_and_embeddings=True)
        return logits, embed
    
class XAptamerBert(nn.Module):
    def __init__(self, cfg):
        super(XAptamerBert, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        
        os.environ["TOKENIZERS_PARALLELISM"] = 'true'

        self.tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_path'])
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,  mlm_probability=cfg['mlm_probability'], return_tensors='pt')
        
        self.x_transformer_encoder= XTransformerEncoder(cfg)
        
    def forward(self, x, attn_mask=None):

        logits, embed = self.x_transformer_encoder(x, mask=attn_mask, return_logits_and_embeddings=True)
        
        return logits, embed
    
class AptamerXTransformerEncoderRegression(nn.Module):
    def __init__(self, cfg):
        super(AptamerXTransformerEncoderRegression, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        
        self.x_transformer_encoder= XTransformerEncoder(cfg)
        self.linear = nn.Linear(self.d_model, 1)

    def forward(self, x, attn_mask=None):

        logits, embed = self.x_transformer_encoder(x, attn_mask=attn_mask)

        x = embed[:, 0, :]

        x = self.linear(x)

        return x
    
        
class DNAXTransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(DNAXTransformerEncoder, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        
        self.x_transformer_encoder= XTransformerEncoder(cfg)
        
        if cfg['model_task'] == 'regression':
            self.linear = nn.Linear(self.d_model, 1)
        
        elif cfg['model_task'] == 'classification' or 'evidence':
            self.linear = nn.Linear(self.d_model, cfg['num_classes'])
            
    def forward(self, x, attn_mask=None):

        logits, embed = self.x_transformer_encoder(x, mask=attn_mask)

        x = embed[:, 0, :]

        x = self.linear(x)

        return x
    
    def predict_uncertainty(self, input_ids, attention_mask, token_type_ids=None):
        y_pred = self(input_ids, attention_mask, token_type_ids)
        
        # dempster-shafer theory
        evidence = relu_evidence(y_pred) # can also try softplus and exp evidence schemes
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        u = self.n_classes / S
        prob = alpha / S
        
        # law of total uncertainty 
        epistemic = prob * (1 - prob) / (S + 1)
        aleatoric = prob - prob**2 - epistemic
        return prob, u, aleatoric, epistemic


class FullDNAXTransformer(nn.Module):
    def __init__(self, cfg):
        super(FullDNAXTransformer, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']

        self.x_transformer = XTransformer(
            dim = cfg['d_model'],
            tie_token_emb = False,
            return_tgt_loss = True,
            enc_num_tokens=cfg['num_tokens'],
            enc_depth = 3,
            enc_heads = 8,
            enc_max_seq_len = cfg['max_seq_len'],
            dec_num_tokens = cfg['num_classes'],
            dec_depth = 3,
            dec_heads = 8,
            dec_max_seq_len = 2
        )
        
    def forward(self, x, y, len_x, scr_mask=None):
        self.batch_size, self.seq_len = x.shape
        
        scr_mask = self.create_mask(len_x)

        x = self.x_transformer(x, y, mask=scr_mask)

        return x
    
    def create_mask(self, len_x):
        mask_pad = torch.zeros(self.batch_size, self.seq_len).bool()
        for (idx, len_comment) in enumerate(len_x): 
            mask_pad[idx, len_comment:self.seq_len] = 1
        return mask_pad.to(self.device)