import torch
import torch.nn.functional as F

from mlguess.torch.class_losses import edl_digamma_loss
from model import *

def get_model(cfg):
    if cfg['model_type'] == "transformer_encoder_classifier":
        return DNATransformerEncoderClassifier(cfg)
    
    elif cfg['model_type'] == "transformer_encoder_regression":
        return DNATransformerEncoderRegression(cfg)
    
    elif cfg['model_type'] == "x_transformer_encoder_regression":
        return AptamerXTransformerEncoderRegression(cfg)
    
    elif cfg['model_type'] == "full_x_transformer":
        return FullDNAXTransformer(cfg)
    
    elif cfg['model_type'] == "aptamer_bert":
        return AptamerBert(cfg)
    
    elif cfg['model_type'] == "aptamer_bert_classifier":
        return AptamerBertClassifier(cfg)
    
    elif cfg['model_type'] == "aptamer_bert_evidence":
        return AptamerBertEvidence(cfg)
    
    elif cfg['model_type'] == "transformer_encoder_evidence":
        return DNATransformerEncoderEvidence(cfg)

    else:
        raise ValueError(f"Invalid model type: {cfg['model_type']}")
    
def get_loss_function(cfg):
    if cfg['model_type'] in (
        'transformer_encoder_regression',
        'x_transformer_encoder_regression'
        ):
        return F.mse_loss
    
    elif cfg['model_type'] in (
        'transformer_encoder_classifier',
        'aptamer_bert',
        'aptamer_bert_classifier'
        ):
        return F.cross_entropy
    
    elif cfg['model_type'] in (
        'aptamer_bert_evidence',
        'transformer_encoder_evidence'
        ):
        return edl_digamma_loss

    else:
        raise ValueError(f"Unknown loss function: {cfg['model_type']}")

def compute_model_output(model, model_inputs, cfg):
    
    if cfg['model_type'] in (
        'transformer_encoder_classifier',
        'transformer_encoder_regression',
        'aptamer_bert_classifier',
        'aptamer_bert_evidence',
        'transformer_encoder_evidence'
        ):
        
        tokenized_seqs = model_inputs[0].to(cfg['device'])
        attn_mask = model_inputs[1].bool()
        attn_mask = ~attn_mask.to(cfg['device'])
                
        output = model(tokenized_seqs, attn_mask=attn_mask)
        return output

    elif cfg['model_type'] in (
        'aptamer_bert'
        ):
        tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_path'])
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,  mlm_probability=cfg['mlm_probability'])
        
        tokenized_data = tokenizer(model_inputs[0], padding=True)
        seqs = tokenized_data['input_ids']
        attn_mask = ~torch.Tensor(tokenized_data['attention_mask']).bool().to(cfg['device'])

        masked_data = data_collator(seqs)
        masked_seqs = masked_data['input_ids'].to(cfg['device'])
        tgt = masked_data['labels'].to(cfg['device'])
        
        logits, embed = model(masked_seqs, attn_mask=attn_mask)
        output = (logits, embed, tgt)
        return output
    
    else:
        raise ValueError(f"Invalid model type: {cfg['model_type']}")

def compute_loss(loss_function, output, target, cfg):
    if cfg['model_type'] in (
        'aptamer_bert_evidence',
        'transformer_encoder_evidence'
        ):
        annealing_coefficient = 10
        target_hot = F.one_hot(target.long(), cfg['num_classes'])
        return loss_function(output, target_hot, cfg['curr_epoch'], cfg['num_classes'], annealing_coefficient, device=cfg['device']) 
        
    elif cfg['model_type'] in (
        'transformer_encoder_classifier',
        'aptamer_bert_classifier'
        ):
        return loss_function(output, target.long().to(cfg['device']))
        
    elif cfg['model_type'] in (
        'transformer_encoder_regression'
        ):
        return loss_function(output.squeeze(), target.float().to(cfg['device']))
        
    elif cfg['model_type'] in (
        'aptamer_bert'
        ):
        src = output[0]
        tgt = output[2]
        return loss_function(src.movedim(2,1), tgt)
        
    else:
        raise ValueError(f"Invalid model type: {cfg['model_type']}")