import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch import optim
import torch.nn.functional as F
from data_utils import load_and_preprocess_data, load_and_preprocess_weighted_frequency_data
from dataset import DNASequenceDataSet
from model import DNATransformerEncoder
from training_utils import train_model, validate_model, test_model
from torch.nn.parallel import DistributedDataParallel
import yaml
import json
import argparse


def initialize_data_and_model(cfg):
    if cfg['load_saved_data_set'] is not False:
        try:
            print("Loading dataset from disk...")
            dna_dataset = torch.load('dna_dataset.pth', map_location=cfg['device'])
        except:
            raise ValueError(f"Invalid dataset file: {cfg['load_saved_data_set']}")
    else:            
        df = load_and_preprocess_weighted_frequency_data(cfg)
        dna_dataset = DNASequenceDataSet(df)
    
    if cfg['save_data_set'] is True:
        torch.save(dna_dataset, 'dna_dataset.pth')
    
    model = get_model(cfg)
    model.to(cfg['device'])
    return dna_dataset, model

def get_model(cfg):
    if cfg['model_type'] == "regression_transformer":
        return DNATransformerEncoder(cfg)
    else:
        raise ValueError(f"Invalid model type: {cfg['model_type']}")

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

    dna_dataset, model = initialize_data_and_model(cfg)

    train_size = int(0.7 * len(dna_dataset))
    val_size = int(0.15 * len(dna_dataset))
    test_size = len(dna_dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dna_dataset, [train_size, val_size, test_size])

    if args.distributed:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, seed=cfg['seed_value'])
        val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, seed=cfg['seed_value'])
        test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, seed=cfg['seed_value'])
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], sampler=train_sampler, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], sampler=val_sampler, num_workers=cfg['num_workers'])
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], sampler=test_sampler, num_workers=cfg['num_workers'])

    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[cfg['device']])
        dist.barrier()

    # Training Loop
    
    loss_dict = {'train_loss': [], 'val_loss': []}
    for epoch in range(cfg['num_epochs']):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        try:
            avg_train_loss, train_loss_list = train_model(model, train_loader, optimizer, cfg)
            avg_val_loss, val_loss_list = validate_model(model, val_loader, cfg)
        except ValueError as e:
            print(f"Error occurred: {e}")
            break
        
        if rank == 0:
            print(f"Epoch: {epoch}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
            
            if args.distributed:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
                
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': state_dict,  # Use model.module.state_dict() for DDP
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint, f"model_checkpoint.pt")
                                
            loss_dict['train_loss'].extend(train_loss_list)
            loss_dict['val_loss'].extend(val_loss_list)
            with open('loss_data.json', 'w') as f:
                json.dump(loss_dict, f)
        
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