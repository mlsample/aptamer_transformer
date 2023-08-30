import torch
import torch.nn.functional as F

def handle_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values.")

def train_model(model, train_loader, optimizer, cfg):
    model.train()
    total_loss = 0
    for batch_idx, (data, target, len_x) in enumerate(train_loader):
        data, target = data.to(cfg['device']), target.to(cfg['device'])
        handle_nan(data, "Training Data")
        handle_nan(target, "Training Target")
        
        optimizer.zero_grad()
        output = model(data, len_x)
        handle_nan(output, "Training Model Output")
        
        loss = F.mse_loss(output.squeeze(), target)
        handle_nan(loss, "Training Loss")
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate_model(model, val_loader, cfg):
    model.eval()
    val_loss = 0.0
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
            
            val_loss += loss.item()
    return val_loss / len(val_loader)


def test_model(model, test_loader, cfg):
    model.eval()
    test_loss = 0.0
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
            
            test_loss += loss.item()
    return test_loss / len(test_loader)