import torch
from tqdm import tqdm
import os
import pickle
import json

from aptamer_transformer.factories_model_loss import get_loss_function, compute_loss, compute_model_output
from aptamer_transformer.metric_utils import *


def train_model(model, train_loader, optimizer, cfg):
    # Set the model to training mode
    model.train()
    
    train_metrics = {'train_loss_list': []}
    total_batches = len(train_loader)
    
    # Get the loss function from the configuration
    loss_function = get_loss_function(cfg)
    
    # Initialize a progress bar for training (only for the main process in distributed training)
    # if (cfg['rank'] == 0) and (cfg['running_echo'] is False):
    if (cfg['rank'] == 0):
        pbar = tqdm(total=total_batches)
        update_interval = 2
    
    # Iterate over each batch in the training data loader
    for batch_idx, (data) in enumerate(train_loader):
        
        # Separate model inputs and target values
        model_inputs, target = data[:-1], data[-1]
        
        # Zero the gradients before backward pass
        optimizer.zero_grad()
        model_outputs = compute_model_output(model, model_inputs, cfg)
        
        # Compute loss and perform backpropagation
        loss = compute_loss(loss_function, model_outputs, target, cfg)
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Append the loss for this batch
        train_metrics['train_loss_list'].append(loss.item())
        train_metrics['avg_train_loss'] = sum(train_metrics['train_loss_list']) / len(train_metrics['train_loss_list'])

        # Update the progress bar
        # if (cfg['rank'] == 0) and (cfg['running_echo'] is False):
        if (cfg['rank'] == 0):
            if batch_idx % update_interval == 0 or batch_idx == total_batches - 1:
                pbar.set_description(f"Batch {batch_idx+1}/{total_batches} | Loss: {train_metrics['avg_train_loss']:.4f}")
                pbar.update(update_interval)
    
    # Close the progress bar after training
    # if (cfg['rank'] == 0) and (cfg['running_echo'] is False):
    if (cfg['rank'] == 0):
        pbar.close()
    
    return train_metrics

def validate_model(model, val_loader, lr_scheduler, cfg):
    model.eval()
    
    val_metrics = {'val_loss_list': [],
                   'y_true_list': [],
                   'y_pred_list': []}
    
    loss_function = get_loss_function(cfg)
    with torch.no_grad():
        for data in val_loader:
            
            model_inputs, target = data[:-1], data[-1]
            
            model_outputs = compute_model_output(model, model_inputs, cfg)
        
            loss = compute_loss(loss_function, model_outputs, target, cfg)

            val_metrics['val_loss_list'].append(loss.item())
            val_metrics['y_true_list'].append(target)
            val_metrics['y_pred_list'].append(model_outputs)
            
    val_metrics['avg_val_loss'] = sum(val_metrics['val_loss_list']) / len(val_metrics['val_loss_list'])
        
    lr_scheduler.step(val_metrics['avg_val_loss'])
    
    if cfg['rank'] == 0:
        
        if cfg['model_config']['learning_task'] == 'regression':
            val_metrics = compute_regression_metrics(val_metrics, cfg)
            
        elif cfg['model_config']['learning_task'] == 'classifier':
            val_metrics = compute_classificaion_metrics(val_metrics, cfg)
            
        elif cfg['model_config']['learning_task'] == 'evidence':
            val_metrics = compute_classificaion_metrics(val_metrics, cfg)
    
    return val_metrics

def test_model(model, test_loader, cfg):
    
    model.eval()
    
    test_metrics = {'test_loss_list': [],
                     'y_true_list': [],
                     'y_pred_list': [],
                     'x_list': []}
    
    loss_function = get_loss_function(cfg)
    
    with torch.no_grad():
        for data in test_loader:
            
            model_inputs, target = data[:-1], data[-1]
            
            model_outputs = compute_model_output(model, model_inputs, cfg)
            loss = compute_loss(loss_function, model_outputs, target, cfg)
            
            test_metrics['test_loss_list'].append(loss.item())
            test_metrics['x_list'].append(model_inputs)
            test_metrics['y_true_list'].append(target)
            test_metrics['y_pred_list'].append(model_outputs)
    
    test_metrics['avg_test_loss'] = sum(test_metrics['test_loss_list']) / len(test_metrics['test_loss_list'])
    
    # Saving y_true and y_pred to a file
    
    with open(f'{cfg["results_path"]}/test_predictions.pkl', 'wb') as f:
        pickle.dump((test_metrics['y_true_list'], test_metrics['y_pred_list']), f)
    with open(f'{cfg["results_path"]}/test_input.pkl', 'wb') as f:
        pickle.dump(test_metrics['x_list'], f)
    
    return test_metrics


def load_checkpoint(model, optimizer, cfg):
    checkpoint = torch.load(cfg['checkpoint_path'], map_location=cfg['device'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    starting_epoch = checkpoint['epoch'] + 1
    
    loss_file = f'{cfg["results_path"]}/metrics.json'
    
    if os.path.exists(loss_file):
        with open(loss_file, 'r') as file:
            metrics = json.load(file)
        
    if cfg['rank'] == 0:
        print('Loaded last checkpoint')
    
    return model, optimizer, starting_epoch, metrics
    

def checkpointing(epoch, train_metrics, val_metrics, args, model, optimizer, metrics, cfg):
    print(f"Epoch: {epoch}, Train Loss: {train_metrics['avg_train_loss']}, Validation Loss: {val_metrics['avg_val_loss']}")
    
    if not os.path.exists(cfg['results_path']):
        os.mkdir(cfg['results_path'])
    
    loss_file = f'{cfg["results_path"]}/metrics.json'
    
    if cfg['running_echo'] is False:
        # Write the updated data back to the file
        with open(loss_file, 'w') as file:
            json.dump(metrics, file, indent=4)

        if args.distributed:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,  # Use model.module.state_dict() for DDP
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_metrics['avg_train_loss'],
            'val_loss': val_metrics['avg_val_loss'],
        }

        if not os.path.exists(cfg['results_path']):
            os.mkdir(cfg['results_path'])

        if os.path.exists(cfg['checkpoint_path']) and cfg['load_last_checkpoint'] is False and epoch == 0:
            os.rename(cfg['checkpoint_path'], cfg['checkpoint_path'] + '.bak' + 'bak')

        elif os.path.exists(cfg['checkpoint_path']):
            os.rename(cfg['checkpoint_path'], cfg['checkpoint_path'] + '.bak')

        torch.save(checkpoint, cfg['checkpoint_path'])
        
    return None


def compute_regression_metrics(val_metrics, cfg):
    
    y_true_flat = np.concatenate(val_metrics['y_true_list'])
    y_pred_flat = np.concatenate([vals.cpu().squeeze().numpy() for vals in val_metrics['y_pred_list']])
    
    mse, rmse, mae, r2, mape, explained_variance = evaluate_regression_model(y_true_flat, y_pred_flat)
    print(f'R2: {r2}, MAPE: {mape}, Explained Variance: {explained_variance}')
    
    val_metrics.update({'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape, 'explained_variance': explained_variance})
    return val_metrics

def compute_classificaion_metrics(val_metrics, cfg):
    
    y_true_flat = np.concatenate(val_metrics['y_true_list'])
    y_pred_flat = np.concatenate([vals.cpu().squeeze() for vals in val_metrics['y_pred_list']])
    
    
    accuracy, precision, recall, fscore, c_matrix, csi = evaluate_classification(y_true_flat, y_pred_flat)
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {fscore}, CSI: {csi}')
    val_metrics.update({'accuracy': accuracy, 'precision': precision, 'recall': recall, 'fscore': fscore, 'csi': csi})
    
    return val_metrics
    

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
        