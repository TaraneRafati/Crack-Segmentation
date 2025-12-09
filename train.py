import torch, torch.nn as nn, torch.optim as optim
from metrics import iou_score, DiceBCELoss
from tqdm import tqdm
from evaluate import evaluate
import os

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, save_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = DiceBCELoss()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training", leave=False)
        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.unsqueeze(1).float().to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            opt.step()
            running_loss += loss.item() * imgs.size(0)
            loop.set_postfix(loss=running_loss / len(train_loader.dataset))
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Validation", leave=False)
            for imgs, masks in val_loop:
                imgs, masks = imgs.to(device), masks.unsqueeze(1).float().to(device)
                out = model(imgs)
                loss = criterion(out, masks)
                val_loss += loss.item() * imgs.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"-> New best model saved with val_loss: {best_val_loss:.4f}")
            
    return history