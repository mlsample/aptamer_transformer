import torch
import torch.nn.functional as F
from tqdm import tqdm

def handle_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values.")

def train_model(model, train_loader, optimizer, cfg):
    model.train()
    train_loss_list = []
    total_batches = len(train_loader)
    
    pbar = tqdm(enumerate(train_loader), total=total_batches)
    
    for batch_idx, (data, target, len_x) in pbar:
        data, target = data.to(cfg['device']), target.to(cfg['device'])
        
        # Check for NaN values
        handle_nan(data, "Training Data")
        handle_nan(target, "Training Target")
        
        optimizer.zero_grad()
        output = model(data, len_x)
        
        # Check for NaN values
        handle_nan(output, "Training Model Output")
        
        loss = F.mse_loss(output.squeeze(), target)
        
        # Check for NaN values
        handle_nan(loss, "Training Loss")
        
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss_list.append(loss.item())
        
        if cfg['rank'] == 1:
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx+1}/{total_batches} completed | Loss: {loss.item():.4f}")
        
    avg_train_loss = sum(train_loss_list) / len(train_loss_list)
    return avg_train_loss, train_loss_list

def validate_model(model, val_loader, cfg):
    model.eval()
    val_loss_list = []
    with torch.no_grad():
        for batch in val_loader:
            x, y, len_x = batch
            x, y = x.to(cfg['device']), y.to(cfg['device'])
            handle_nan(x, "Validation Data")
            handle_nan(y, "Validation Target")
            
            output = model(x, len_x)
            handle_nan(output, "Validation Model Output")
            
            loss = F.mse_loss(output.squeeze(), y)
            handle_nan(loss, "Validation Loss")
            
            val_loss_list.append(loss.item())
            
    avg_val_loss = sum(val_loss_list) / len(val_loss_list)
    return avg_val_loss, val_loss_list

def test_model(model, test_loader, cfg):
    model.eval()
    test_loss_list = []
    with torch.no_grad():
        for batch in test_loader:
            x, y, len_x = batch
            handle_nan(x, "Test Data")
            handle_nan(y, "Test Target")
            
            x, y = x.to(cfg['device']), y.to(cfg['device'])
            
            output = model(x, len_x)
            handle_nan(output, "Test Model Output")
            
            loss = F.mse_loss(output.squeeze(), y)
            handle_nan(loss, "Test Loss")
            
            test_loss_list.append(loss.item())
    
    avg_test_loss = sum(test_loss_list) / len(test_loss_list)
    return avg_test_loss, test_loss_list