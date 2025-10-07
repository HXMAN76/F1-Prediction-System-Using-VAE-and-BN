# Quick test to verify VAE implementation works
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import glob
import os

print("🧠 Testing VAE Implementation")
print(f"🔥 PyTorch version: {torch.__version__}")
print(f"🎯 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# Load the most recent preprocessed data
data_dir = "data/preprocessed"
if os.path.exists(data_dir):
    pattern = os.path.join(data_dir, "vae_input_df_*.csv")
    files = glob.glob(pattern)
    if files:
        latest_file = max(files, key=os.path.getctime)
        print(f"📁 Loading data from: {latest_file}")
        vae_df = pd.read_csv(latest_file)
        print(f"✅ Data loaded successfully: {vae_df.shape}")
        print(f"📊 Features: {list(vae_df.columns)}")
        print("\n🔍 Data preview:")
        print(vae_df.head())
        
        # Check for any non-numeric columns
        numeric_cols = vae_df.select_dtypes(include=[np.number]).columns
        print(f"\n🔢 Numeric columns ({len(numeric_cols)}): {list(numeric_cols)}")
        non_numeric_cols = vae_df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"⚠️ Non-numeric columns ({len(non_numeric_cols)}): {list(non_numeric_cols)}")
    else:
        print("❌ No VAE input files found")
else:
    print("❌ Preprocessed data directory not found")