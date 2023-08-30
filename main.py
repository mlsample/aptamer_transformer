import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch import optim
import torch.nn.functional as F
from data_utils import load_and_preprocess_data
from dataset import DNASequenceDataSet
from model import DNATransformerEncoder
from training_utils import train_model, validate_model, test_model
from torch.nn.parallel import DistributedDataParallel

def initialize_data_and_model(cfg):
    directory = '/scratch/mlsample/protein_dnn/sars-cov-2-data/processed'
    df = load_and_preprocess_data(directory)
    if cfg['debug'] is True:
        df = df.head(1000)
    dna_dataset = DNASequenceDataSet(df)
    model = DNATransformerEncoder(cfg)
    model.to(cfg['device'])
    return dna_dataset, model


def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)    

    cfg = {
        'device': device,
        'num_layers': 6,
        'd_model': 2,
        'nhead': 1,
        'd_ff': 2,
        'dropout_rate': 0.1,
        'num_epochs': 10,
        'batch_size': 64,
        'learning_rate': 0.001,
        'debug': False,
    }

    dna_dataset, model = initialize_data_and_model(cfg)

    train_size = int(0.7 * len(dna_dataset))
    val_size = int(0.15 * len(dna_dataset))
    test_size = len(dna_dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dna_dataset, [train_size, val_size, test_size])

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], sampler=val_sampler)
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], sampler=test_sampler)

    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    model = DistributedDataParallel(model, device_ids=[cfg['device']])

    # Training Loop
    for epoch in range(cfg['num_epochs']):
        train_sampler.set_epoch(epoch)
        
        try:
            train_loss = train_model(model, train_loader, optimizer, cfg)
            val_loss = validate_model(model, val_loader, cfg)
        except ValueError as e:
            print(f"Error occurred: {e}")
            break
        
        if rank == 0:
            print(f"Epoch: {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
    
    # Test the model after training
    try:
        test_loss = test_model(model, test_loader, cfg)
        if rank == 0:
            print(f"Test Loss: {test_loss}")
    except ValueError as e:
        print(f"Error occurred during testing: {e}")

if __name__ == "__main__":
    main()