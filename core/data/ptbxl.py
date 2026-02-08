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
    
    def __init__(self, data_path, sampling_rate=100, batch_size=32):
        """
        Args:
            data_path: Path to PTB-XL dataset
            sampling_rate: Target sampling rate (100 or 500 Hz)
            batch_size: Batch size for saving signals
        """
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
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
    
    def save_in_batches(self, data, split_name):
        """
        Save signals and labels in batches to organized folder structure.
        
        Args:
            data: Signal data array (N, channels, time)
            labels: Label data array (N, num_classes)
            split_name: 'train', 'validation', or 'test'
        """
        # Create directories
        signals_dir = os.path.join('data', 'signals', split_name)
        # labels_dir = os.path.join('data', 'labels', split_name)
        os.makedirs(signals_dir, exist_ok=True)
        # os.makedirs(labels_dir, exist_ok=True)
        
        # Calculate number of batches
        num_samples = len(data)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        
        print(f"Saving {split_name} data in {num_batches} batches (batch_size={self.batch_size})")
        
        # Save batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_samples)
            
            batch_data = data[start_idx:end_idx]
            # batch_labels = labels[start_idx:end_idx]
            
            # Save batch files
            np.save(os.path.join(signals_dir, f'batch_{batch_idx}.npy'), batch_data)
            # np.save(os.path.join(labels_dir, f'batch_{batch_idx}.npy'), batch_labels)
    
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
        
        # Save processed data in batches
        # print("\nSaving processed data...")
        # self.save_in_batches(X_train, 'train')
        # self.save_in_batches(X_val, 'validation')
        # self.save_in_batches(X_test, 'test')
        
        # Save labels split wise
        np.save(f"{self.data_path}/labels/y_train.py", y_train)
        np.save(f"{self.data_path}/labels/y_val.py", y_val)
        np.save(f"{self.data_path}/labels/y_test.py", y_test)
        
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
    processor = PTBXLProcessor(DATA_PATH, sampling_rate=100, batch_size=32)
    data = processor.preprocess()
    
    print("\nData loading and processing done!")