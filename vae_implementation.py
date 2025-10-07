import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import glob
import os
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("🧠 F1 Probability Simulator - VAE Implementation")
print(f"🔥 PyTorch version: {torch.__version__}")
print(f"🎯 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load data
data_dir = "data/preprocessed"
pattern = os.path.join(data_dir, "vae_input_df_*.csv")
files = glob.glob(pattern)
latest_file = max(files, key=os.path.getctime)
print(f"📁 Loading data from: {latest_file}")

vae_df = pd.read_csv(latest_file)
print(f"✅ Data loaded successfully: {vae_df.shape}")

# Separate features and target
X = vae_df.drop('target', axis=1)
y = vae_df['target']

print(f"📊 Features shape: {X.shape}")
print(f"🎯 Target shape: {y.shape}")
print(f"🔢 Features: {list(X.columns)}")

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None  # Can't stratify continuous target
)

print(f"📈 Training set: {X_train.shape}")
print(f"📊 Test set: {X_test.shape}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train.values)
X_test_tensor = torch.FloatTensor(X_test.values)
y_train_tensor = torch.FloatTensor(y_train.values)
y_test_tensor = torch.FloatTensor(y_test.values)

print("\n🚀 Data prepared for VAE training!")

# VAE Architecture
class F1_VAE(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=16, latent_dim=4):
        super(F1_VAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim//2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Position predictor from latent space
        self.position_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 1)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def predict_position(self, z):
        return self.position_predictor(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        position_pred = self.predict_position(z)
        return reconstructed, mu, logvar, z, position_pred

# Loss function
def vae_loss_function(recon_x, x, mu, logvar, pred_pos, true_pos, 
                     recon_weight=1.0, kl_weight=0.5, position_weight=2.0):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Position prediction loss
    position_loss = F.mse_loss(pred_pos.squeeze(), true_pos, reduction='mean')
    
    # Combined loss
    total_loss = (recon_weight * recon_loss + 
                 kl_weight * kl_loss + 
                 position_weight * position_loss)
    
    return total_loss, recon_loss, kl_loss, position_loss

# Initialize model
input_dim = X_train.shape[1]
model = F1_VAE(input_dim=input_dim, hidden_dim=16, latent_dim=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\n🏗️ VAE Model initialized:")
print(f"   Input dim: {input_dim}")
print(f"   Hidden dim: 16")
print(f"   Latent dim: 4")
print(f"   Parameters: {sum(p.numel() for p in model.parameters())}")

# Training
print("\n🏋️ Training VAE...")
model.train()
epochs = 100
train_losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass
    recon_batch, mu, logvar, z, pred_pos = model(X_train_tensor)
    
    # Calculate loss
    loss, recon_loss, kl_loss, pos_loss = vae_loss_function(
        recon_batch, X_train_tensor, mu, logvar, pred_pos, y_train_tensor
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: Total={loss.item():.4f} | "
              f"Recon={recon_loss.item():.4f} | "
              f"KL={kl_loss.item():.4f} | "
              f"Position={pos_loss.item():.4f}")

print("\n✅ Training completed!")

# Evaluation
model.eval()
with torch.no_grad():
    # Test set evaluation
    recon_test, mu_test, logvar_test, z_test, pred_pos_test = model(X_test_tensor)
    
    test_loss, test_recon, test_kl, test_pos = vae_loss_function(
        recon_test, X_test_tensor, mu_test, logvar_test, pred_pos_test, y_test_tensor
    )
    
    # Position prediction accuracy
    pos_mae = F.l1_loss(pred_pos_test.squeeze(), y_test_tensor).item()
    pos_rmse = torch.sqrt(F.mse_loss(pred_pos_test.squeeze(), y_test_tensor)).item()
    
    print(f"\n📊 Test Results:")
    print(f"   Total Loss: {test_loss.item():.4f}")
    print(f"   Reconstruction Loss: {test_recon.item():.4f}")
    print(f"   KL Loss: {test_kl.item():.4f}")
    print(f"   Position Loss: {test_pos.item():.4f}")
    print(f"   Position MAE: {pos_mae:.4f}")
    print(f"   Position RMSE: {pos_rmse:.4f}")

# Generate latent representations for entire dataset
print("\n🔗 Generating latent representations for Bayesian Network integration...")
model.eval()
with torch.no_grad():
    X_full_tensor = torch.FloatTensor(X.values)
    _, mu_full, _, z_full, pred_pos_full = model(X_full_tensor)
    
    # Use mean (mu) as deterministic latent representation
    latent_vectors = mu_full.numpy()

# Create integration dataset
latent_df = pd.DataFrame(latent_vectors, columns=[f'latent_dim_{i}' for i in range(4)])
latent_df['actual_position'] = y.values
latent_df['predicted_position'] = pred_pos_full.squeeze().numpy()

print(f"🎯 Latent representation created: {latent_df.shape}")
print("\n🔍 Latent space preview:")
print(latent_df.head())

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "data/vae_results"
os.makedirs(output_dir, exist_ok=True)

# Save model
model_path = os.path.join(output_dir, f"f1_vae_model_{timestamp}.pth")
torch.save(model.state_dict(), model_path)
print(f"\n💾 Model saved: {model_path}")

# Save latent representations
latent_path = os.path.join(output_dir, f"latent_vectors_{timestamp}.csv")
latent_df.to_csv(latent_path, index=False)
print(f"💾 Latent vectors saved: {latent_path}")

# Save training summary
summary = {
    'model_architecture': {
        'input_dim': input_dim,
        'hidden_dim': 16,
        'latent_dim': 4,
        'total_parameters': sum(p.numel() for p in model.parameters())
    },
    'training_config': {
        'epochs': epochs,
        'learning_rate': 0.001,
        'final_loss': train_losses[-1]
    },
    'performance': {
        'test_total_loss': test_loss.item(),
        'test_reconstruction_loss': test_recon.item(),
        'test_kl_loss': test_kl.item(),
        'test_position_loss': test_pos.item(),
        'position_mae': pos_mae,
        'position_rmse': pos_rmse
    },
    'data_info': {
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': list(X.columns)
    }
}

import json
summary_path = os.path.join(output_dir, f"vae_training_summary_{timestamp}.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"💾 Training summary saved: {summary_path}")

print(f"\n🎉 VAE Implementation Complete!")
print(f"📈 Final training loss: {train_losses[-1]:.4f}")
print(f"🎯 Position prediction RMSE: {pos_rmse:.4f} positions")
print(f"🔗 Ready for Bayesian Network integration!")