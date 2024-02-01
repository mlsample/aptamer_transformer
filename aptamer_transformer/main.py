import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import traceback

from aptamer_transformer.training_utils import train_model, validate_model, test_model, checkpointing, load_checkpoint, EarlyStopper
from aptamer_transformer.data_utils import load_dataset, get_data_loaders, read_cfg
from aptamer_transformer.distributed_utils import ddp_setup_process_group
from aptamer_transformer.factories_model_loss import get_model, get_lr_scheduler, model_config


def parse_arguments():
    parser = argparse.ArgumentParser(description='Aptamer Transformer Training Script')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    return parser.parse_known_args()[0]

def main():
    args = parse_arguments()
    cfg = read_cfg(args.config)
    metrics = train_and_evaluate(cfg, args=args)
    
    return metrics

def train_and_evaluate(cfg, trial=None, args=None):
    
    # Set the seed for generating random numbers
    torch.manual_seed(cfg['seed_value'])
    torch.cuda.manual_seed_all(cfg['seed_value'])
    
    # Setup for distributed data parallel (DDP) training if required
    rank, cfg = ddp_setup_process_group(cfg, args.distributed)

    # Load the dataset
    dna_dataset = load_dataset(cfg)

    # Initialize the model
    model = get_model(cfg)
    model.to(cfg['device'])

    # Get data loaders for training, validation, and testing
    train_loader, val_loader, test_loader, train_sampler = get_data_loaders(dna_dataset, cfg, args)
    
    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    # Set up the learning rate scheduler
    lr_scheduler = get_lr_scheduler(optimizer, cfg)
    
    # Load last checkpoint if required
    if cfg['load_last_checkpoint'] is True:
        model, optimizer, starting_epoch, metrics = load_checkpoint(model, optimizer, cfg)
    else:
        metrics = {'train_loss': [], 'val_loss': []}
        starting_epoch = 0
    
    # Wrap the model for distributed training
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[cfg['device']], find_unused_parameters=True)
        dist.barrier()

    early_stopper = EarlyStopper(patience=cfg['stopping_patience'], min_delta=cfg['min_delta']) 
    # Main training loop
    for epoch in range(starting_epoch, cfg['num_epochs'] + starting_epoch):
        cfg['curr_epoch'] = epoch
        
        # Set the epoch for the distributed sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        try:
            # Train the model for one epoch
            train_metrics = train_model(model, train_loader, optimizer, cfg)
            
            # Validate the model
            val_metrics = validate_model(model, val_loader, lr_scheduler, cfg)
            
            # if early_stopper.early_stop(val_metrics['avg_val_loss']): 
            #     print('Early stopping at epoch: ', epoch)            
            #     break
            
        except Exception as e:
            # Print the exception traceback if an error occurs
            print(traceback.format_exc())
            raise e
 
        # Save checkpoints and log losses
        metrics['train_loss'].append(train_metrics['train_loss_list'])
        metrics['val_loss'].append(val_metrics['val_loss_list'])
        if rank == 0:
            checkpointing(epoch, train_metrics, val_metrics, args, model, optimizer, metrics, cfg)
        
        # Barrier for distributed training
        if args.distributed:
            dist.barrier()
            
    # Test the model after training is complete
    try:
        test_metrics = test_model(model, test_loader, cfg)
        
        # Log test loss
        if rank == 0:
            print(f"Test Loss: {test_metrics['avg_test_loss']}")
            
    except ValueError as e:
        print(traceback.format_exc())
        raise e

    return metrics
    
if __name__ == "__main__":
    main()