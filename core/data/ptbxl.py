"""
PTB-XL ECG Data Loader and Preprocessor
"""

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os


class PTBXLProcessor:
    """Load and preprocess PTB-XL dataset"""
    
    def __init__(self, data_path, sampling_rate=100):
        """
        Args:
            data_path: Path to PTB-XL dataset
            sampling_rate: Target sampling rate (100 or 500 Hz)
        """
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.leads = [0, 1, 7]  # I, II, V2 as per paper
        
        # 4 CVD classes from paper
        self.classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        
    def load_raw_data(self, df, sampling_rate):
        """Load raw ECG signals"""
        if sampling_rate == 100:
            data = [wfdb.rdsamp(os.path.join(self.data_path, f)) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(os.path.join(self.data_path, f)) for f in df.filename_hr]
        
        data = np.array([signal for signal, meta in data])
        return data
    
    def prepare_labels(self, df):
        """Prepare multi-label targets"""
        # Map diagnostic superclass to binary labels
        labels = np.zeros((len(df), len(self.classes)))
        
        for idx, row in df.iterrows():
            scp_codes = row['scp_codes']
            
            # Check each class
            if 'NORM' in scp_codes:
                labels[idx, 0] = 1
            if 'MI' in scp_codes:
                labels[idx, 1] = 1
            if 'STTC' in scp_codes:
                labels[idx, 2] = 1
            if 'CD' in scp_codes:
                labels[idx, 3] = 1
            if 'HYP' in scp_codes:
                labels[idx, 4] = 1
        
        return labels
    
    def preprocess(self):
        """Main preprocessing pipeline"""
        print("Loading PTB-XL dataset...")
        
        # Load database
        df = pd.read_csv(os.path.join(self.data_path, 'ptbxl_database.csv'))
        df.scp_codes = df.scp_codes.apply(lambda x: eval(x))
        
        # Load raw signals
        print(f"Loading raw data at {self.sampling_rate}Hz...")
        data = self.load_raw_data(df, self.sampling_rate)
        
        # Select specific leads (I, II, V2)
        print(f"Selecting leads: I, II, V2")
        data = data[:, :, self.leads]  # Shape: (N, time, 3)
        data = np.transpose(data, (0, 2, 1))  # Shape: (N, 3, time)
        
        # Prepare labels
        print("Preparing multi-label targets...")
        labels = self.prepare_labels(df)
        
        # Get train/val/test split
        print("Creating train/val/test splits...")
        train_fold = df[df.strat_fold <= 8]
        val_fold = df[df.strat_fold == 9]
        test_fold = df[df.strat_fold == 10]
        
        train_idx = train_fold.index.values
        val_idx = val_fold.index.values
        test_idx = test_fold.index.values
        
        # Split data
        X_train = data[train_idx]
        y_train = labels[train_idx]
        
        X_val = data[val_idx]
        y_val = labels[val_idx]
        
        X_test = data[test_idx]
        y_test = labels[test_idx]
        
        # Normalize (z-score normalization)
        print("Normalizing signals...")
        scaler = StandardScaler()
        
        # Reshape for scaling
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        
        # Fit on train, transform all
        scaler.fit(X_train_reshaped)
        
        X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_val = scaler.transform(X_val_reshaped).reshape(X_val.shape)
        X_test = scaler.transform(X_test_reshaped).reshape(X_test.shape)
        
        print(f"\nDataset Statistics:")
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Val: {X_val.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")
        print(f"Signal shape: {X_train.shape[1:]} (leads, time)")
        print(f"Classes: {self.classes}")
        print(f"Class distribution (train):")
        for i, cls in enumerate(self.classes):
            print(f"  {cls}: {y_train[:, i].sum():.0f} ({y_train[:, i].mean()*100:.1f}%)")
        
        # Save processed data
        print("\nSaving processed data...")
        os.makedirs('processed', exist_ok=True)
        
        np.save('processed/X_train.npy', X_train)
        np.save('processed/y_train.npy', y_train)
        np.save('processed/X_val.npy', X_val)
        np.save('processed/y_val.npy', y_val)
        np.save('processed/X_test.npy', X_test)
        np.save('processed/y_test.npy', y_test)
        
        os.makedirs('saved', exist_ok=True)
        
        with open('saved/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        print("Data preprocessing complete!")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test),
            'scaler': scaler
        }


if __name__ == "__main__":
    # Example usage
    DATA_PATH = "X:\ecg_fm\data"
    
    # Preprocess data
    processor = PTBXLProcessor(DATA_PATH, sampling_rate=100)
    data = processor.preprocess()
    
    print("\nData loading and processing done!")