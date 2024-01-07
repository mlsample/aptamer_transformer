import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle
from mlguess.torch.class_losses import edl_digamma_loss
import json
import torch.distributed as dist

def get_loss_function(cfg):
    if cfg['model_task'] == 'regression':
        return F.mse_loss
    elif cfg['model_task'] == 'evidence':
        return edl_digamma_loss
    elif cfg['model_task'] == 'classification':
        return F.cross_entropy
    elif cfg['model_task'] == 'mlm':
        return F.cross_entropy
    else:
        raise ValueError(f"Unknown loss function: {cfg['model_task']}")

def compute_loss(loss_function, output, target, cfg):
    if cfg['model_task'] == 'evidence':
        annealing_coefficient = 10
        target_hot = F.one_hot(target.long(), cfg['num_classes'])
        loss = loss_function(output, target_hot, cfg['curr_epoch'], cfg['num_classes'], annealing_coefficient, device=cfg['device']) 
        
    elif cfg['model_task'] == 'classification':
        loss = loss_function(output, target.long())
        
    elif cfg['model_task'] == 'regression':
        loss = loss_function(output.squeeze(), target.float()) 
        
    elif cfg['model_task'] == 'mlm':
        src = output[0]
        tgt = output[2]
        loss = loss_function(src.movedim(2,1), tgt)
    
    return loss

def train_model(model, train_loader, optimizer, cfg):
    model.train()
    train_loss_list = []
    total_batches = len(train_loader)
    
    if cfg['rank'] == 0:
        pbar = tqdm(total=total_batches)
    update_interval = 2
    loss_function = get_loss_function(cfg)
    
    for batch_idx, (data, target, len_x) in enumerate(train_loader):
        if cfg['model_type'] != 'aptamer_bert':
            data, target = data.to(cfg['device']), target.to(cfg['device'])

            # Check for NaN values
            handle_nan(data, "Training Data")
            handle_nan(target, "Training Target")
        
        optimizer.zero_grad()
        
        output = model(data, len_x)
        
        # Check for NaN values
        # handle_nan(output, "Training Model Output")
        
        loss = compute_loss(loss_function, output, target, cfg)
        
        # Check for NaN values
        handle_nan(loss, "Training Loss")
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
    
    loss_function = get_loss_function(cfg)
    with torch.no_grad():
        for batch in val_loader:
            x, y, len_x = batch
            if cfg['model_type'] != 'aptamer_bert':
                x, y = x.to(cfg['device']), y.to(cfg['device'])

                # Check for NaN values
                handle_nan(x, "Training Data")
                handle_nan(y, "Training Target")
            
            output = model(x, len_x)
            # handle_nan(output, "Validation Model Output")
            
            # if cfg['model_type'] != 'x_transformer_encoder':
            #     output = output.squeeze()    

            loss = compute_loss(loss_function, output, y, cfg)
            handle_nan(loss, "Validation Loss")

            val_loss_list.append(loss.item())
            
    avg_val_loss = sum(val_loss_list) / len(val_loss_list)
    lr_scheduler.step(avg_val_loss)
    return avg_val_loss, val_loss_list

def test_model(model, test_loader, cfg):
    model.eval()
    test_loss_list = []
    y_true_list = []
    y_pred_list = []
    
    loss_function = get_loss_function(cfg)
    with torch.no_grad():
        for batch in test_loader:
            x, y, len_x = batch
            if cfg['model_type'] != 'aptamer_bert':
                x, y = x.to(cfg['device']), y.to(cfg['device'])

                # Check for NaN values
                handle_nan(x, "Training Data")
                handle_nan(y, "Training Target")
            
            output = model(x, len_x)
            # handle_nan(output, "Test Model Output")
            
            # if cfg['model_type'] != 'x_transformer_encoder':
            #     output = output.squeeze()    
            
            loss = compute_loss(loss_function, output, y, cfg)
            handle_nan(loss, "Test Loss")
            
            test_loss_list.append(loss.item())
            
            if cfg['model_task'] == 'mlm':
                y_true_list.append(x)
            else:
                y_true_list.append(y.cpu().numpy())
                
            if cfg['model_task'] == 'mlm':
                out = output[0].cpu().numpy()
                tgt = output[1].cpu().numpy()
                y_pred_list.append(((out, tgt)))
            else:
                y_pred_list.append(output.cpu().numpy())
    
    avg_test_loss = sum(test_loss_list) / len(test_loss_list)
    
    # Saving y_true and y_pred to a file
    
    with open(f'{cfg["results_path"]}/test_predictions.pkl', 'wb') as f:
        pickle.dump((y_true_list, y_pred_list), f)
    
    return avg_test_loss, test_loss_list


def handle_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values.")

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
    torch.save(checkpoint, cfg['checkpoint_path'])
                        
    loss_dict['train_loss'].extend(train_loss_list)
    loss_dict['val_loss'].extend(val_loss_list)
    with open(f'{cfg["results_path"]}/loss_data.json', 'w') as f:
        json.dump(loss_dict, f)

    return None
        