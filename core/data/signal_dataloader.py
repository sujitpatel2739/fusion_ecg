import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from .precission_cache import PrecisionCache

class ECGSignalDataset(Dataset):
    """PyTorch Dataset for PTB-XL ECG data"""
    
    def __init__(self, signals_path, labels_path, saved_batch_size, transform=None):
        """
        Args:
            signals: Path to directory containing image batches (e.g., 'data/signals/train/')
            labels_path: Path to labels .npy file
            saved_batch_size: Number of samples per saved batch file (default: 32)
            transform: Optional transforms to apply
        """
        # data can be either raw signals or precomputed images depending on the model 
        self.image_path = signals_path
        self.labels = np.load(labels_path)
        self.saved_batch_size = saved_batch_size
        self.transform = transform
        self.cache = PrecisionCache(capacity=50)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Calculate which batch file and position within that batch
        batch_no = idx // self.saved_batch_size
        sample_idx = idx % self.saved_batch_size
        
        # Try to get from cache
        batch = self.cache.get_batch(batch_no)
        
        # Load from disk if not in cache
        if batch is None:
            batch_file = os.path.join(self.image_path, f"batch_{batch_no}.npy")
            
            if not os.path.exists(batch_file):
                raise FileNotFoundError(f"Batch file not found: {batch_file}")
            
            batch = np.load(batch_file)
            self.cache.add_batch(batch_no, batch)
        
        # Handle last batch edge case
        if sample_idx >= len(batch):
            raise IndexError(
                f"Sample index {sample_idx} out of bounds for batch {batch_no} "
                f"with {len(batch)} samples"
            )
        
        # Get image and label
        signal = batch[sample_idx].astype(np.float32)
        label = self.labels[idx].astype(np.float32)
        
        # Convert to tensors
        signal = torch.from_numpy(signal).float()
        label = torch.from_numpy(label).float()
        
        # Apply transforms if any
        if self.transform:
            signal = self.transform(signal)
        
        return signal, label
    
def create_signal_dataloader(
    image_path,
    labels_path,
    saved_batch_size=32,
    batch_size=16,  # DataLoader batch size (can be different from saved_batch_size)
    shuffle=True,
    num_workers=4
):
    """
    Create dataloader for image data
    
    Args:
        signals_path: Path to Signals directory
        labels_path: Path to labels file
        saved_batch_size: Size of saved batch files (default: 32)
        batch_size: DataLoader batch size (default: 16)
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    """
    dataset = ECGSignalDataset(
        signals_path=image_path,
        labels_path=labels_path,
        saved_batch_size=saved_batch_size,
        transform=None
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory= torch.cuda.is_available()
    )
    
    return loader

