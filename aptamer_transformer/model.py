import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from x_transformers import XTransformer, TransformerWrapper, Decoder, Encoder
from aptamer_transformer.load_foundation_models import load_seq_struct_x_aptamer_transformer

os.environ["TOKENIZERS_PARALLELISM"] = 'true'
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from mlguess.torch.class_losses import relu_evidence, softplus_evidence, exp_evidence
    

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
    

class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']

        self.embed = nn.Embedding(cfg['num_tokens'], self.d_model, padding_idx=0)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg['num_layers'])])
        
    def forward(self, x, attn_mask=None):
        self.batch_size, self.seq_len = x.shape
        
        x = self.embed(x)
        x = x + self.get_position_encoding(self.seq_len, self.d_model).to(self.device)
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        return x
    
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
        return torch.FloatTensor(pos_enc)
    
class TransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super(TransformerDecoder, self).__init__()
        self.cfg = cfg
        self.d_model = cfg['d_model']
        
        self.embed = nn.Embedding(cfg['num_tokens'], self.d_model, padding_idx=0)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg['num_layers'])])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        self.batch_size, self.seq_len = x.shape
        
        x = self.embed(x)
        x = x + self.get_position_encoding(self.seq_len, self.d_model).to(self.cfg['device'])
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return x
    
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
        return torch.FloatTensor(pos_enc)

class SeqStructEnerMatrixRegression(nn.Module):
    def __init__(self, cfg):
        super(SeqStructEnerMatrixRegression, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        
        self.linear1 = nn.Linear(1, cfg['d_model'])
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg['num_layers'])])
        self.linear2 = nn.Linear(cfg['d_model'], 1)
        self.linear3 = nn.Linear(cfg['max_seq_len'], 1)
        
    def forward(self, x, attn_mask=None):
        x = self.linear1(x)
        
        for layer in self.layers:
            x = layer(x, attn_mask=None)
        
        x = self.linear2(x)
        x = x.squeeze()
        
        x = self.linear3(x)
        
        return x
    

class TransformerEncoderClassifier(nn.Module):
    def __init__(self, cfg):
        super(TransformerEncoderClassifier, self).__init__()
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
    

class TransformerEncoderRegression(nn.Module):
    def __init__(self, cfg):
        super(TransformerEncoderRegression, self).__init__()
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
    

class TransformerEncoderEvidence(TransformerEncoderClassifier):
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


class DNATransformerEncoderClassifier(TransformerEncoderClassifier):
    pass


class DNATransformerEncoderEvidence(TransformerEncoderEvidence):
    pass
    
    
class DNATransformerEncoderRegression(TransformerEncoderRegression):
    pass
    
    
class StructTransformerEncoderClassifier(TransformerEncoderClassifier):
    pass
    

class StructTransformerEncoderRegression(TransformerEncoderRegression):
    pass
    
    
class SeqStructTransformerEncoderRegression(TransformerEncoderRegression):
    pass

class SeqStructTransformerEncoderClassifier(TransformerEncoderClassifier):
    pass

    
class SeqStructAptamerBert(nn.Module):
    def __init__(self, cfg):
        super(SeqStructAptamerBert, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        
        os.environ["TOKENIZERS_PARALLELISM"] = 'true'

        self.tokenizer = AutoTokenizer.from_pretrained(cfg['seq_struct_tokenizer_path'])
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,  mlm_probability=cfg['mlm_probability'], return_tensors='pt')
        
        self.transformer_encoder = TransformerEncoder(cfg)
        self.linear = nn.Linear(self.d_model, cfg['num_tokens'])
        
    def forward(self, x, attn_mask=None):

        embed = x = self.transformer_encoder(x, attn_mask=attn_mask)
        
        logits = self.linear(x)
        
        return logits, embed
    
class AptamerBert(nn.Module):
    def __init__(self, cfg):
        super(AptamerBert, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        
        os.environ["TOKENIZERS_PARALLELISM"] = 'true'

        self.tokenizer = AutoTokenizer.from_pretrained(cfg['seq_tokenizer_path'])
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
    
class AptamerBertRegression(nn.Module):
    def __init__(self, cfg):
        super(AptamerBertRegression, self).__init__()

        self.aptamer_bert_encoding = AptamerBert(cfg)
        self.aptamer_bert_encoding.load_state_dict(
            torch.load(cfg['aptamer_bert_path'], map_location=cfg['device']
                )['model_state_dict']
            )
        
        self.linear = nn.Linear(cfg['d_model'], 1)
        
    def forward(self, x, attn_mask=None):
        logits, embed = self.aptamer_bert_encoding(x, attn_mask=attn_mask)
        
        x = embed[:, 0, :]
        
        return self.linear(x)

class XTransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super(XTransformerEncoder, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['seq_struct_tokenizer_path'])
        cfg['num_tokens'] = self.tokenizer.vocab_size
        cfg['max_seq_len'] = self.tokenizer.model_max_length
        
        self.x_transformer_encoder= TransformerWrapper(
            num_tokens = cfg['num_tokens'],
            max_seq_len = cfg['max_seq_len'],
            num_memory_tokens = cfg['num_memory_tokens'],
            l2norm_embed = cfg['l2norm_embed'],
            emb_dropout=cfg['dropout_rate'],
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
                attn_talking_heads = cfg['attn_talking_heads'],
                attn_gate_values = cfg['attn_gate_values'],
                macaron = cfg['macaron'],
                rel_pos_bias = cfg['rel_pos_bias'],
                rotary_pos_emb = cfg['rotary_pos_emb'],
                rotary_xpos = cfg['rotary_xpos'],
                residual_attn = cfg['residual_attn'],
                pre_norm = cfg['pre_norm'],
                attn_qk_norm = cfg['attn_qk_norm'],
                attn_qk_norm_dim_scale = cfg['attn_qk_norm_dim_scale'],
            )
        )

    def forward(self, x, attn_mask=None):
        logits, embed = self.x_transformer_encoder(x, mask=~attn_mask, return_logits_and_embeddings=True)
        return logits, embed
    
class XTransformerEncoderClassifier(nn.Module):
    def __init__(self, cfg):
        super(XTransformerEncoderClassifier, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        
        self.x_transformer_encoder= XTransformerEncoder(cfg)
        self.linear = nn.Linear(self.d_model, cfg['num_classes'])
        
    def forward(self, x, attn_mask=None):

        logits, embed = self.x_transformer_encoder(x, attn_mask=attn_mask)

        x = embed[:, 0, :]

        x = self.linear(x)

        return x
    
class XTransformerEncoderEvidence(nn.Module):
    def __init__(self, cfg):
        super(XTransformerEncoderEvidence, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        self.n_classes = cfg['num_classes']
        
        self.x_transformer_encoder= XTransformerEncoder(cfg)
        self.linear = nn.Linear(self.d_model, cfg['num_classes'])
        
    def forward(self, x, attn_mask=None):

        logits, embed = self.x_transformer_encoder(x, attn_mask=attn_mask)

        x = embed[:, 0, :]

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
    
class XTransformerEncoderRegression(nn.Module):
    def __init__(self, cfg):
        super(XTransformerEncoderRegression, self).__init__()
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
    

class XAptamerBert(nn.Module):
    def __init__(self, cfg):
        super(XAptamerBert, self).__init__()
        self.cfg = cfg
        self.device = cfg['device']
        self.d_model = cfg['d_model']
        
        os.environ["TOKENIZERS_PARALLELISM"] = 'true'
        
        self.x_transformer_encoder= XTransformerEncoder(cfg)
        
    def forward(self, x, attn_mask=None):

        logits, embed = self.x_transformer_encoder(x, attn_mask=attn_mask)
        
        return logits, embed
    
class SeqStructXAptamerBert(XAptamerBert):
    pass
    
    
class SeqStructXAptamerBertRegression(nn.Module):
    def __init__(self, cfg):
        super(SeqStructXAptamerBertRegression, self).__init__()

        self.aptamer_bert_encoding = load_seq_struct_x_aptamer_transformer(cfg)

        
        self.linear = nn.Linear(cfg['d_model'], 1)
        
    def forward(self, x, attn_mask=None):
        logits, embed = self.aptamer_bert_encoding(x, attn_mask=attn_mask)
        
        x = embed[:, 0, :]
        
        return self.linear(x), embed
    

class SeqStructXAptamerBertClassifier(nn.Module):
    def __init__(self, cfg):
        super(SeqStructXAptamerBertClassifier, self).__init__()

        self.aptamer_bert_encoding = load_seq_struct_x_aptamer_transformer(cfg)
        
        self.linear_0 = nn.Linear(120, cfg['d_model'])
        
        self.x_transformer_encoder= XTransformerEncoder(cfg)
        
        self.linear_1 = nn.Linear(cfg['d_model'], cfg['num_classes'])
        
    def forward(self, x, attn_mask=None):
        logits, x = self.aptamer_bert_encoding(x, attn_mask=attn_mask)
        
        x = self.linear_0(x)
        
        logits, x = self.x_transformer_encoder(x, attn_mask=attn_mask)
        
        x = x[:, 0, :]
        
        return self.linear(x)
    
    
class SeqStructXAptamerBertEvidence(nn.Module):
    def __init__(self, cfg):
        super(SeqStructXAptamerBertEvidence, self).__init__()

        self.aptamer_bert_encoding = load_seq_struct_x_aptamer_transformer(cfg)
        
        self.linear = nn.Linear(cfg['d_model'], cfg['num_classes'])
        
        
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
        prob = MinMaxScaler().fit_transform(prob)
        
        # law of total uncertainty 
        epistemic = prob * (1 - prob) / (S + 1)
        aleatoric = prob - prob**2 - epistemic
        return prob, u, aleatoric, epistemic



class XAptamerBertClassifier(nn.Module):
    def __init__(self, cfg):
        super(XAptamerBertClassifier, self).__init__()

        self.aptamer_bert_encoding = XAptamerBert(cfg)
        self.aptamer_bert_encoding.load_state_dict(
            torch.load(cfg['x_aptamer_bert_path'], map_location=cfg['device']
                )['model_state_dict']
            )
        
        self.linear = nn.Linear(cfg['d_model'], cfg['num_classes'])
        
    def forward(self, x, attn_mask=None):
        logits, embed = self.aptamer_bert_encoding(x, attn_mask=attn_mask)
        
        x = embed[:, 0, :]
        
        return self.linear(x)
    
class XAptamerBertEvidence(nn.Module):
    def __init__(self, cfg):
        super(XAptamerBertEvidence, self).__init__()

        self.aptamer_bert_encoding = XAptamerBert(cfg)
        self.aptamer_bert_encoding.load_state_dict(
            torch.load(cfg['x_aptamer_bert_path'], map_location=cfg['device']
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
    
class XAptamerBertRegression(nn.Module):
    def __init__(self, cfg):
        super(XAptamerBertRegression, self).__init__()

        self.aptamer_bert_encoding = XAptamerBert(cfg)
        self.aptamer_bert_encoding.load_state_dict(
            torch.load(cfg['x_aptamer_bert_path'], map_location=cfg['device']
                )['model_state_dict']
            )
        
        self.linear = nn.Linear(cfg['d_model'], 1)
        
    def forward(self, x, attn_mask=None):
        logits, embed = self.aptamer_bert_encoding(x, attn_mask=attn_mask)
        
        x = embed[:, 0, :]
        
        return self.linear(x)

    
class MinMaxScaler:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val
        self.data_min = None
        self.data_max = None

    def fit(self, X):
        self.data_min = X.min(dim=0)[0]
        self.data_max = X.max(dim=0)[0]

    def transform(self, X):
        X_scaled = (X - self.data_min) / (self.data_max - self.data_min)
        X_scaled = X_scaled * (self.max_val - self.min_val) + self.min_val
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)