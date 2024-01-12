import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

import argparse
import traceback

from training_utils import train_model, validate_model, test_model, checkpointing, load_checkpoint
from data_utils import load_dataset, get_data_loaders, read_cfg
from distributed_utils import ddp_setup_process_group
from factories_model_loss import get_model, get_lr_scheduler


def parse_arguments():
    parser = argparse.ArgumentParser(description='Aptamer occurance regression task')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    return parser.parse_known_args()[0]

def main():
    
    args = parse_arguments()
    cfg = read_cfg(args.config)
    
    torch.manual_seed(cfg['seed_value'])
    torch.cuda.manual_seed_all(cfg['seed_value'])
    
    rank, cfg = ddp_setup_process_group(cfg, args.distributed)

    dna_dataset = load_dataset(cfg)
    model = get_model(cfg)
    model.to(cfg['device'])

    train_loader, val_loader, test_loader, train_sampler = get_data_loaders(dna_dataset, cfg, args)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    lr_scheduler = get_lr_scheduler(optimizer, cfg)
    
    if cfg['load_last_checkpoint'] is True:
        model, optimizer, starting_epoch, loss_dict = load_checkpoint(model, optimizer, cfg)
    else:
        loss_dict = {'train_loss': [], 'val_loss': []}
        starting_epoch = 0
    
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[cfg['device']], find_unused_parameters=True)
        dist.barrier()

    # Training Loop
    for epoch in range(starting_epoch, cfg['num_epochs'] + starting_epoch):
        cfg['curr_epoch'] = epoch
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        try:
            avg_train_loss, train_loss_list = train_model(model, train_loader, optimizer, cfg)
            
            avg_val_loss, val_loss_list = validate_model(model, val_loader, lr_scheduler, cfg)
            
        except Exception:
            return print(traceback.format_exc())
 
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
        return print(traceback.format_exc())

if __name__ == "__main__":
    main()