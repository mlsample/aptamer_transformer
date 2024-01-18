import torch
import torch.nn.functional as F

from mlguess.torch.class_losses import edl_digamma_loss
from aptamer_transformer.model import *
from aptamer_transformer.dataset import *


def model_config(cfg):
    MODEL_CONFIG = {
        ############################
        # Regression Models
        ############################
        "transformer_encoder_regression": {
            "class": DNATransformerEncoderRegression,
            "learning_task": "regression",
            "dataset_class": SeqRegressionDataset,
        },
        "aptamer_bert_regression": {
            "class": AptamerBertRegression,
            "learning_task": "regression",
            "dataset_class": SeqRegressionDataset,
        },
        "x_transformer_encoder_regression": {
            "class": XTransformerEncoderRegression,
            "learning_task": "regression",
            "dataset_class": SeqRegressionDataset,
        },
        "x_aptamer_bert_regression": {
            "class": XAptamerBertRegression,
            "learning_task": "regression",
            "dataset_class": SeqRegressionDataset,
        },
        "struct_transformer_encoder_regression": {
            "class": StructTransformerEncoderRegression,
            "learning_task": "regression",
            "dataset_class": StructRegressionDataset,
        },
        'seq_struct_transformer_encoder_regression': {
            "class": SeqStructTransformerEncoderRegression,
            "learning_task": "regression",
            "dataset_class": SeqStructRegressionDataSet,
        },

        ############################
        # Classification Models
        ############################
        "transformer_encoder_classifier": {
            "class": DNATransformerEncoderClassifier,
            "learning_task": "classifier",
            "dataset_class": SeqClassifierDataset,
        },
        "struct_transformer_encoder_classifier": {
            "class": StructTransformerEncoderClassifier,
            "learning_task": "classifier",
            "dataset_class": StructClassifierDataset,
        },
        "aptamer_bert_classifier": {
            "class": AptamerBertClassifier,
            "learning_task": "classifier",
            "dataset_class": SeqClassifierDataset,
        },
        "x_transformer_encoder_classifier": {
            "class": XTransformerEncoderClassifier,
            "learning_task": "classifier",
            "dataset_class": SeqClassifierDataset,
        },
        "x_aptamer_bert_classifier": {
            "class": XAptamerBertClassifier,
            "learning_task": "classifier",
            "dataset_class": SeqClassifierDataset,
        },
        ############################
        # Evidence Models
        ############################
        "transformer_encoder_evidence": {
            "class": DNATransformerEncoderEvidence,
            "learning_task": "evidence",
            "dataset_class": SeqClassifierDataset,
        },
        "aptamer_bert_evidence": {
            "class": AptamerBertEvidence,
            "learning_task": "evidence",
            "dataset_class": SeqClassifierDataset,
        },
        "x_transformer_encoder_evidence": {
            "class": XTransformerEncoderEvidence,
            "learning_task": "evidence",
            "dataset_class": SeqClassifierDataset,
        },
        "x_aptamer_bert_evidence": {
            "class": XAptamerBertEvidence,
            "learning_task": "evidence",
            "dataset_class": SeqClassifierDataset,
        },

        ############################
        # Masked Language Models
        ############################
        "aptamer_bert": {
            "class": AptamerBert,
            "learning_task": "masked_language_model",
            "dataset_class": SeqBertDataSet,
        },
        "x_aptamer_bert": {
            "class": XAptamerBert,
            "learning_task": "masked_language_model",
            "dataset_class": SeqBertDataSet,
        },
        "seq_struct_aptamer_bert": {
            "class": SeqStructAptamerBert,
            "learning_task": "masked_language_model",
            "dataset_class": SeqStructBertDataSet,
        },
        
    }
    
    if cfg['model_type'] in MODEL_CONFIG:
        cfg['model_config'] = MODEL_CONFIG[cfg['model_type']]
        
    else:
        raise ValueError(f"Invalid model type: {cfg['model_type']}")
    
    return cfg

def get_model(cfg):
    model_config = cfg['model_config']
    if 'class' in model_config:
        model_class = model_config["class"]
        return model_class(cfg)
    else:
        raise ValueError(f"Invalid model config: {model_config}")
    
 
def get_loss_function(cfg):
    learning_task = cfg['model_config']['learning_task']
    
    ############################
    # Regression Models
    ############################
    if learning_task == 'regression':
        return F.mse_loss
    
    ###############################################
    # Classification and Masked Language Models
    ###############################################
    elif learning_task in ('classifier', 'masked_language_model'):
        return F.cross_entropy
    
    ############################
    # Evidence Models
    ############################
    elif learning_task == 'evidence':
        return edl_digamma_loss

    else:
        raise ValueError(f"Invalid learning task: {learning_task}")

def compute_model_output(model, model_inputs, cfg):
    learning_task = cfg['model_config']['learning_task']
    ####################################################
    # Classification, Evidence, and Regression Models
    ####################################################
    if learning_task in ('classifier', 'evidence', 'regression'):
        
        tokenized_seqs = model_inputs[0].to(cfg['device'])
        attn_mask = model_inputs[1].bool()
        attn_mask = ~attn_mask.to(cfg['device'])
                
        output = model(tokenized_seqs, attn_mask=attn_mask)
        return output

    ############################
    # Masked Language Models
    ############################
    elif learning_task == 'masked_language_model':
        
        tokenizer = AutoTokenizer.from_pretrained(cfg['seq_struct_tokenizer_path'])
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
        raise ValueError(f"Invalid learning task: {learning_task}")

def compute_loss(loss_function, output, target, cfg):
    learning_task = cfg['model_config']['learning_task']
    
    ############################
    # Evidence Models
    ############################
    if learning_task == 'evidence':
        annealing_coefficient = 10
        target_hot = F.one_hot(target.long(), cfg['num_classes'])
        return loss_function(output, target_hot, cfg['curr_epoch'], cfg['num_classes'], annealing_coefficient, device=cfg['device']) 
    
    ############################
    # Classification Models
    ############################
    elif learning_task == 'classifier':
        return loss_function(output, target.long().to(cfg['device']))
    
    ############################
    # Regression Models
    ############################
    elif learning_task == 'regression':
        return loss_function(output.squeeze(), target.float().to(cfg['device']))
    
    ############################
    # Masked Language Models
    ############################
    elif learning_task == 'masked_language_model':
        src = output[0]
        tgt = output[2]
        return loss_function(src.movedim(2,1), tgt)
        
    else:
        raise ValueError(f"Invalid learning task: {learning_task}")
    
    
def get_lr_scheduler(optimizer, cfg):
    if cfg['lr_scheduler'] == 'ReduceLROnPlateau':
        if cfg['rank'] == 0:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                patience = cfg['lr_patience'], 
                verbose = True,
                min_lr = 1.0e-13
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                patience = cfg['lr_patience'], 
                verbose = False,
                min_lr = 1.0e-13
            )
        return lr_scheduler
    
    if cfg['lr_scheduler'] == 'CosineAnnealingWarmRestarts':
        if cfg['rank'] == 0:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0  = cfg['lr_patience'], 
                verbose = True,
                eta_min  = 1.0e-13
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0 = cfg['lr_patience'], 
                verbose = False,
                eta_min  = 1.0e-13
            )
        return lr_scheduler
    
    else:
        raise ValueError(f"Invalid lr_scheduler: {cfg['lr_scheduler']}")