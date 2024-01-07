import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch import optim
import torch.nn.functional as F
from data_utils import load_dataset, get_data_loaders
from dataset import DNASequenceDataSet
from model import get_model
from training_utils import train_model, validate_model, test_model, checkpointing
from torch.nn.parallel import DistributedDataParallel
import yaml
import json
import argparse
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description='Aptamer occurance regression task')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    return parser.parse_known_args()[0]

def main():
    args = parse_arguments()
    
    # Read the YAML configuration file
    with open(args.config, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    torch.manual_seed(cfg['seed_value'])
    torch.cuda.manual_seed_all(cfg['seed_value'])
    
    if args.distributed:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        cfg['is_distributed'] = True
    else:
        rank = 0
        world_size = 1

    device = torch.device("cuda", rank)    

    # Update the cfg dictionary with dynamic values
    cfg.update({
        'device': device,
        'rank': rank,
        'world_size': world_size
    })

    dna_dataset = load_dataset(cfg)
    model = get_model(cfg)
    model.to(cfg['device'])

    train_loader, val_loader, test_loader, train_sampler = get_data_loaders(dna_dataset, cfg, args)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    if rank == 0:
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
    
    if cfg['load_last_checkpoint'] is True:
        checkpoint = torch.load(cfg['checkpoint_path'], map_location=cfg['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if rank == 0:
            print('Loaded last checkpoint')
    
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[cfg['device']], find_unused_parameters=True)
        dist.barrier()

    # Training Loop
    
    loss_dict = {'train_loss': [], 'val_loss': []}
    for epoch in range(cfg['num_epochs']):
        cfg['curr_epoch'] = epoch
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        try:
            avg_train_loss, train_loss_list = train_model(model, train_loader, optimizer, cfg)
            avg_val_loss, val_loss_list = validate_model(model, val_loader, lr_scheduler, cfg)
        except ValueError as e:
            print(f"Error occurred: {e}")
            break
 
        if rank == 0:
            checkpointing(epoch, avg_train_loss, avg_val_loss, args, model, optimizer, loss_dict, train_loss_list, val_loss_list, cfg)
        
        if args.distributed:
            dist.barrier()
            
    # Test the model after training
    try:
        avg_test_loss, test_loss_list = test_model(model, test_loader, cfg)
        if rank == 0:
            print(f"Test Loss: {avg_test_loss}")
    except ValueError as e:
        print(f"Error occurred during testing: {e}")

if __name__ == "__main__":
    main()