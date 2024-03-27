from aptamer_transformer.data_utils import read_cfg

import torch

def load_seq_struct_x_aptamer_transformer(cfg):
    from aptamer_transformer.model import SeqStructXAptamerBert
    seq_struct_loader_path = cfg['seq_struct_aptamer_bert_path']
    
    seq_struct_cfg = read_cfg(f'{seq_struct_loader_path}/config.yaml', new_working_dir=cfg['working_dir'])
    
    seq_struct_cfg.update({
    'device': cfg['device'],
    })
    
    seq_struct_model = SeqStructXAptamerBert(seq_struct_cfg)
    
    seq_struct_model_checkpoint = torch.load(f'{seq_struct_loader_path}/seq_struct_x_aptamer_bert_checkpoint.pt', map_location=cfg['device'])
    del seq_struct_model_checkpoint['model_state_dict']['x_transformer_encoder.x_transformer_encoder.to_logits.bias']
    
    seq_struct_model.load_state_dict(seq_struct_model_checkpoint['model_state_dict'])
    
    return seq_struct_model