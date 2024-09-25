import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.path_names import WEIGHTS_PATH, TRAIN_HISTORY

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import build_vit_model
import pickle
from tqdm import tqdm

def train_model(train_dataset, val_dataset, batch_size, num_epochs=20):
    """
    Trains the Visual Transformer model on the given data.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of training epochs.

    Returns:
        tuple: Trained model and training history.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_vit_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_correct += ((outputs > 0) == labels.unsqueeze(1)).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float().unsqueeze(1))
                
                val_loss += loss.item() * inputs.size(0)
                val_correct += ((outputs > 0) == labels.unsqueeze(1)).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    torch.save(model.state_dict(), WEIGHTS_PATH + 'vit_weights.pth')
    
    with open(TRAIN_HISTORY + "HistoryViTDict", "wb") as file_pi:
        pickle.dump(history, file_pi)
    
    return model, history

def define_model():
    """
    Defines and loads a pre-trained Visual Transformer model.

    Returns:
        VisualTransformerModel: Loaded pre-trained model.
    """
    model = build_vit_model()
    model.load_state_dict(torch.load(WEIGHTS_PATH + 'vit_weights.pth'))
    return model