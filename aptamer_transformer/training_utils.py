import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle
from mlguess.torch.class_losses import edl_digamma_loss
import json
import torch.distributed as dist
from factories_model_loss import get_loss_function, compute_loss, compute_model_output
from metric_utils import *
import os

def train_model(model, train_loader, optimizer, cfg):
    model.train()
    train_loss_list = []
    total_batches = len(train_loader)
    
    loss_function = get_loss_function(cfg)
    
    if cfg['rank'] == 0:
        pbar = tqdm(total=total_batches)
        update_interval = 2
    
    for batch_idx, (data) in enumerate(train_loader):
        
        model_inputs, target = data[:-1], data[-1]
        
        optimizer.zero_grad()
        model_outputs = compute_model_output(model, model_inputs, cfg)
        
        loss = compute_loss(loss_function, model_outputs, target, cfg)
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss_list.append(loss.item())
        
        if cfg['rank'] == 0:  # Update every 10 batches
            # Update the progress bar every 'update_interval' steps
            if batch_idx % update_interval == 0 or batch_idx == total_batches - 1:
                avg_loss = sum(train_loss_list) / len(train_loss_list)
                pbar.set_description(f"Batch {batch_idx+1}/{total_batches} | Loss: {avg_loss:.4f}")
                pbar.update(update_interval)
        

    if cfg['rank'] == 0:
        pbar.close()
    avg_train_loss = sum(train_loss_list) / len(train_loss_list)
    
    return avg_train_loss, train_loss_list

def validate_model(model, val_loader, lr_scheduler, cfg):
    model.eval()
    val_loss_list = []
    y_true_list = []
    y_pred_list = []
    
    loss_function = get_loss_function(cfg)
    with torch.no_grad():
        for data in val_loader:
            
            model_inputs, target = data[:-1], data[-1]
            
            model_outputs = compute_model_output(model, model_inputs, cfg)
        
            loss = compute_loss(loss_function, model_outputs, target, cfg)

            val_loss_list.append(loss.item())
            y_true_list.append(target)
            y_pred_list.append(model_outputs)
            
    avg_val_loss = sum(val_loss_list) / len(val_loss_list)
    lr_scheduler.step(avg_val_loss)
    # if cfg['rank'] == 0 and 'writer' in cfg:
    #     compute_classificaion_metrics(y_pred_list, y_true_list, cfg)
    return avg_val_loss, val_loss_list

def test_model(model, test_loader, cfg):
    
    model.eval()
    
    test_loss_list = []
    x_list = []
    y_true_list = []
    y_pred_list = []
    
    loss_function = get_loss_function(cfg)
    
    with torch.no_grad():
        for data in test_loader:
            
            model_inputs, target = data[:-1], data[-1]
            
            model_outputs = compute_model_output(model, model_inputs, cfg)
            loss = compute_loss(loss_function, model_outputs, target, cfg)
            
            test_loss_list.append(loss.item())
            x_list.append(model_inputs)
            y_true_list.append(target)
            y_pred_list.append(model_outputs)
    
    avg_test_loss = sum(test_loss_list) / len(test_loss_list)
    
    # Saving y_true and y_pred to a file
    
    with open(f'{cfg["results_path"]}/test_predictions.pkl', 'wb') as f:
        pickle.dump((y_true_list, y_pred_list), f)
    with open(f'{cfg["results_path"]}/test_input.pkl', 'wb') as f:
        pickle.dump(x_list, f)
    
    return avg_test_loss, test_loss_list


def load_checkpoint(model, optimizer, cfg):
    checkpoint = torch.load(cfg['checkpoint_path'], map_location=cfg['device'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    starting_epoch = checkpoint['epoch'] + 1
    
    loss_file = f'{cfg["results_path"]}/loss_data.json'
    
    if os.path.exists(loss_file):
        with open(loss_file, 'r') as file:
            loss_dict = json.load(file)
        
    if cfg['rank'] == 0:
        print('Loaded last checkpoint')
    
    return model, optimizer, starting_epoch, loss_dict
    

def checkpointing(epoch, avg_train_loss, avg_val_loss, args, model, optimizer, loss_dict, train_loss_list, val_loss_list, cfg):
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
    
    if not os.path.exists(cfg['results_path']):
        os.mkdir(cfg['results_path'])
    
    if os.path.exists(cfg['checkpoint_path']) and cfg['load_last_checkpoint'] is False and epoch == 0:
        os.rename(cfg['checkpoint_path'], cfg['checkpoint_path'] + '.bak' + 'bak')

    elif os.path.exists(cfg['checkpoint_path']):
        os.rename(cfg['checkpoint_path'], cfg['checkpoint_path'] + '.bak')

    torch.save(checkpoint, cfg['checkpoint_path'])
                        
    loss_file = f'{cfg["results_path"]}/loss_data.json'
    
    # Append new loss data
    loss_dict['train_loss'].append(train_loss_list)
    loss_dict['val_loss'].append(val_loss_list)
    
    # Write the updated data back to the file
    with open(loss_file, 'w') as file:
        json.dump(loss_dict, file, indent=4)
        
    return None


def compute_classificaion_metrics(y_preds, y_true, cfg):
    y_true = torch.cat(y_true, dim=0)
    y_preds = torch.cat(y_preds, dim=0)
    
    y_true = y_true.cpu().numpy()
    y_preds = y_preds.cpu().numpy()
    
    y_preds = np.argmax(y_preds, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_preds, average='weighted')
    
    if cfg['rank'] == 0 and 'writer' in cfg:
        cfg['writer'].add_scalar('Precision', precision, cfg['curr_epoch'])
        cfg['writer'].add_scalar('Recall', recall, cfg['curr_epoch'])
        cfg['writer'].add_scalar('F1', f1, cfg['curr_epoch'])
    
    return None
    

        