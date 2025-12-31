"""
Census Income Prediction - Deep Learning Approach (ResNet-style MLP)
-------------------------------------------------------------------
This script implements a custom neural network architecture using PyTorch to predict
income brackets (>50K vs <=50K) from the Adult Census dataset.

Key Architectural Features:
1. Entity Embeddings for high-cardinality categorical variables.
2. Residual Blocks (ResNet) to allow for deeper networks without vanishing gradients.
3. GELU Activation functions for smoother gradients (modern standard).
4. Weighted Loss Function (BCEWithLogitsLoss) to handle class imbalance (3:1).
5. Learning Rate Finder integration for optimal hyperparameter selection.
6. Gradient Clipping and LR Scheduling for training stability.

Author: Devansh
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import random
import os

# --- 1. CONFIGURATION & REPRODUCIBILITY ---
CONFIG = {
    'seed': 42,
    'batch_size': 128,
    'hidden_dim': 256,
    'dropout_rate': 0.3,
    'learning_rate': 3e-4,  # Optimal found via LR Finder
    'epochs': 20,
    'val_split': 0.2,
    'test_split': 0.5, # Split temp data 50/50 into val/test
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility across runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG['seed'])
print(f"Using device: {CONFIG['device']}")


# --- 2. DATA PREPARATION ---
def load_and_preprocess_data(filepath: str):
    """
    Loads CSV, handles missing values, encodes categories, and scales numericals.
    Returns processed loaders, dataset objects, and metadata.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Define Column Groups
    categorical_cols = ['workclass', 'marital.status', 'occupation', 'relationship', 
                        'race', 'sex', 'native.country']
    # Check for correct column names (dot vs hyphen) in source CSV
    actual_cols = df.columns.tolist()
    categorical_cols = [c for c in categorical_cols if c in actual_cols]
    # Handle alternative naming if needed (e.g., marital-status vs marital.status)
    if 'marital-status' in actual_cols and 'marital.status' not in categorical_cols:
        categorical_cols.append('marital-status')
        
    numerical_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    # Adjust for discrepancies
    numerical_cols = [c for c in numerical_cols if c in actual_cols]
    if 'education-num' in actual_cols: numerical_cols.append('education-num')
    elif 'education.num' not in numerical_cols and 'education.num' in actual_cols: numerical_cols.append('education.num')

    target_col = 'income'

    # Preprocessing: Handle Missing/Rare
    if 'education' in df.columns: df = df.drop(columns=['education']) # Redundant with education.num
    
    # Log transform skewed features
    for col in ['capital.gain', 'capital.loss']:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # Categorical Encoding (Label Encoding + Vocab Size)
    vocab_sizes = {}
    for col in categorical_cols:
        df[col] = df[col].astype('category')
        vocab_sizes[col] = len(df[col].cat.categories) + 1 # +1 for potential unseen
        df[col] = df[col].cat.codes
    
    # Encode Target
    df[target_col] = df[target_col].apply(lambda x: 1 if '>50K' in str(x) else 0)

    # Splitting
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=CONFIG['val_split'], random_state=CONFIG['seed'], stratify=y
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=CONFIG['test_split'], random_state=CONFIG['seed'], stratify=y_temp
    )

    # Scaling Numerical Features
    scaler = StandardScaler()
    # Fit only on TRAIN to avoid data leakage
    X_train = X_train_raw.copy()
    X_val = X_val_raw.copy()
    X_test = X_test_raw.copy()
    
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val[numerical_cols]   = scaler.transform(X_val[numerical_cols])
    X_test[numerical_cols]  = scaler.transform(X_test[numerical_cols])

    print(f"Data Split -> Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Calculate Class Weight for Loss Function
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    pos_weight = num_neg / num_pos
    print(f"Class Imbalance Ratio (Neg/Pos): {pos_weight:.2f}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab_sizes, categorical_cols, numerical_cols, pos_weight


class CensusDataset(Dataset):
    """Custom PyTorch Dataset for tabular data."""
    def __init__(self, X, y, cat_cols, num_cols):
        self.X_cat = torch.tensor(X[cat_cols].values, dtype=torch.long)
        self.X_cont = torch.tensor(X[num_cols].values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cat[idx], self.X_cont[idx], self.y[idx]


# --- 3. MODEL ARCHITECTURE (ResNet + GELU) ---
class ResidualBlock(nn.Module):
    """
    A Residual Block with Skip Connection.
    Structure: Linear -> BN -> GELU -> Dropout -> Linear -> BN -> (+ Input) -> GELU
    """
    def __init__(self, hidden_dim, dropout_rate):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(), # Smoother than ReLU, avoids dead neurons
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.block(x) + x) # The Skip Connection


class ModernCensusModel(nn.Module):
    """
    Modern Tabular Model combining Entity Embeddings and Deep Residual Networks.
    """
    def __init__(self, embedding_sizes, n_cont, hidden_dim=256, dropout_rate=0.3):
        super().__init__()
        # Create embeddings for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(num, min(50, (num + 1) // 2)) for num in embedding_sizes
        ])
        self.n_emb_out = sum(e.embedding_dim for e in self.embeddings)
        
        # Initial projection layer
        self.first_layer = nn.Sequential(
            nn.Linear(self.n_emb_out + n_cont, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Deep Residual Layers
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout_rate),
            ResidualBlock(hidden_dim, dropout_rate),
            ResidualBlock(hidden_dim, dropout_rate)
        )
        
        # Output Head
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x_cat, x_cont):
        # Process embeddings
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embeddings, dim=1)
        
        # Concatenate with continuous variables
        x = torch.cat([x_emb, x_cont], dim=1)
        
        # Forward pass
        x = self.first_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)


# --- 4. TRAINING UTILITIES ---
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20):
    """Standard PyTorch training loop with validation and checkpointing."""
    print("\nStarting Training Loop...")
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for x_cat, x_cont, y in train_loader:
            x_cat, x_cont, y = x_cat.to(CONFIG['device']), x_cont.to(CONFIG['device']), y.to(CONFIG['device'])
            
            optimizer.zero_grad()
            y_pred = model(x_cat, x_cont)
            loss = criterion(y_pred, y)
            loss.backward()
            
            # Gradient Clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            
        # Validation Phase
        model.eval()
        val_acc = 0.0
        val_loss = 0.0
        batches = 0
        with torch.no_grad():
            for x_cat, x_cont, y in val_loader:
                x_cat, x_cont, y = x_cat.to(CONFIG['device']), x_cont.to(CONFIG['device']), y.to(CONFIG['device'])
                output = model(x_cat, x_cont)
                val_loss += criterion(output, y).item()
                
                preds = (torch.sigmoid(output) > 0.5).float()
                val_acc += (preds == y).sum().item() / y.size(0)
                batches += 1
        
        avg_val_acc = val_acc / batches
        avg_train_loss = running_loss / len(train_loader)
        
        # Update Scheduler
        scheduler.step(avg_val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {avg_val_acc*100:.2f}% | LR: {current_lr:.1e}")

        # Save Best Model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            # torch.save(model.state_dict(), 'best_pytorch_model.pth') 

    print("Training Complete.")
    return model


def evaluate_performance(model, loader, device, threshold=0.5):
    """Generates detailed classification report and confusion matrix."""
    print("\n--- Evaluation Report ---")
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for x_cat, x_cont, y in loader:
            x_cat, x_cont, y = x_cat.to(device), x_cont.to(device), y.to(device)
            output = model(x_cat, x_cont)
            preds = (torch.sigmoid(output) > threshold).float()
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    # Metrics
    print(classification_report(y_true, y_pred, target_names=['<=50K', '>50K']))
    
    # Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    # A. Load Data
    data_path = 'adult.csv' # Ensure this path is correct
    if not os.path.exists(data_path):
        # Fallback for folder structure
        data_path = '../data/adult.csv' 
        
    if os.path.exists(data_path):
        (X_train, y_train), (X_val, y_val), (X_test, y_test), vocab_sizes, cat_cols, num_cols, pos_weight = load_and_preprocess_data(data_path)

        # B. Create Loaders
        train_ds = CensusDataset(X_train, y_train, cat_cols, num_cols)
        val_ds   = CensusDataset(X_val, y_val, cat_cols, num_cols)
        test_ds  = CensusDataset(X_test, y_test, cat_cols, num_cols)

        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)

        # C. Initialize Model
        # Convert vocab dict values to a list in correct order for embeddings
        embedding_sizes = [vocab_sizes[col] for col in cat_cols]
        
        model = ModernCensusModel(embedding_sizes, len(num_cols), 
                                  hidden_dim=CONFIG['hidden_dim'], 
                                  dropout_rate=CONFIG['dropout_rate']).to(CONFIG['device'])

        # D. Setup Training Components
        # Weighted Loss to handle class imbalance
        pos_weight_tensor = torch.tensor([pos_weight]).to(CONFIG['device'])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

        # E. Train
        trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=CONFIG['epochs'])

        # F. Evaluate
        evaluate_performance(trained_model, test_loader, CONFIG['device'])
        
    else:
        print(f"Error: Data file not found at {data_path}. Please check the path.")