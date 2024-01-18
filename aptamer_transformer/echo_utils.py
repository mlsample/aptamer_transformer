from echo.src.base_objective import BaseObjective
import optuna

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

import argparse
import traceback

from aptamer_transformer.training_utils import train_model, validate_model, test_model, checkpointing, load_checkpoint
from aptamer_transformer.data_utils import load_dataset, get_data_loaders, read_cfg
from aptamer_transformer.distributed_utils import ddp_setup_process_group
from aptamer_transformer.factories_model_loss import get_model, get_lr_scheduler, model_config

class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss", device="cpu"):
        # Initialize the base class
        BaseObjective.__init__(self, config, metric, device)
    def train(self, trial, conf):
        try:
            return main(conf, trial=trial)
        except Exception as e:
            if "CUDA" in str(e):
                optuna.TrialPrune()
            else:
                raise print(traceback.format_exc())
    
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Aptamer occurance regression task')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    return parser.parse_known_args()[0]

def main(cfg, trial=False):
    
    # Parse command-line arguments
    args = parse_arguments()

    # Read configuration from YAML file
    cfg = model_config(cfg)
    working_dir = cfg['working_dir']
    model_type = cfg['model_type']
    cfg = {k: v.replace('{WORKING_DIR}', f'{working_dir}') if isinstance(v, str) else v for k, v in cfg.items()}
    cfg = {k: v.replace('{MODEL_TYPE}', f'{model_type}') if isinstance(v, str) else v for k, v in cfg.items()}
           
    
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
        model, optimizer, starting_epoch, loss_dict = load_checkpoint(model, optimizer, cfg)
    else:
        loss_dict = {'train_loss': [], 'val_loss': []}
        starting_epoch = 0
    
    # Wrap the model for distributed training
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[cfg['device']], find_unused_parameters=True)
        dist.barrier()

    # Main training loop
    for epoch in range(starting_epoch, cfg['num_epochs'] + starting_epoch):
        cfg['curr_epoch'] = epoch
        
        # Set the epoch for the distributed sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        try:
            # Train the model for one epoch
            avg_train_loss, train_loss_list = train_model(model, train_loader, optimizer, cfg)
            
            # Validate the model
            avg_val_loss, val_loss_list = validate_model(model, val_loader, lr_scheduler, cfg)
            
        except Exception:
            # Print the exception traceback if an error occurs
            return print(traceback.format_exc())
 
        # Save checkpoints and log losses
        if rank == 0:
            checkpointing(epoch, avg_train_loss, avg_val_loss, args, model, optimizer, loss_dict, train_loss_list, val_loss_list, cfg)
        
        # Barrier for distributed training
        if args.distributed:
            dist.barrier()
            
    # Test the model after training is complete
    try:
        avg_test_loss, test_loss_list = test_model(model, test_loader, cfg)
        
        # Log test loss
        if rank == 0:
            print(f"Test Loss: {avg_test_loss}")
            
    except ValueError as e:
        return print(traceback.format_exc())
    
    avg_val_loss = np.mean(np.array(loss_dict['val_loss']), axis=1)
    
    best_fold = [
        i for i, j in enumerate(avg_val_loss) if j == min(avg_val_loss)
    ][0]
    
    return {"val_loss": avg_val_loss[best_fold]}

if __name__ == "__main__":
    main()