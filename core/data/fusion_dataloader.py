"""
DataLoader for ECG Images (GAF/MTF)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from .precission_cache import PrecisionCache


class ECGFusionDataset(Dataset):
    """
    Dataset for loading batched ECG Signalss + images
    """
    def __init__(self, signal_path, image_path, labels_path, saved_batch_size=32, transform_type='gaf', transform=None):
        """
        Args:
            image_path: Path to directory containing image batches (e.g., 'data/images/train/')
            signal_path: Path to directory containing signals batches (e.g., 'data/signals/train/')
            labels_path: Path to labels .npy file
            saved_batch_size: Number of samples per saved batch file (default: 32)
            transform_type: 'gaf' or 'mtf'
            transform: Optional transforms to apply
        """
        self.image_path = os.path.join(image_path, transform_type)
        self.signal_path = os.path.join(signal_path)
        self.labels = np.load(labels_path)
        self.saved_batch_size = saved_batch_size
        self.transform_type = transform_type
        self.transform = transform
        self.cache = PrecisionCache(capacity=50)
        
        # Verify directories exists
        if not os.path.exists(self.image_path):
            raise ValueError(f"Image directory not found: {self.image_path}")
        if not os.path.exists(self.signal_path):
            raise ValueError(f"Signal directory not found: {self.signal_path}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Calculate which batch file and position within that batch
        batch_no = idx // self.saved_batch_size
        sample_idx = idx % self.saved_batch_size
        
        # Try to get from cache
        signal_batch = self.cache.get_batch(f"signal_b_{batch_no}")
        image_batch = self.cache.get_batch(f"image_b_{batch_no}")
        
        # Load from disk if not in cache
        if signal_batch is None:
            batch_file = os.path.join(self.signal_path, f"batch_{batch_no}.npy")
            
            if not os.path.exists(batch_file):
                raise FileNotFoundError(f"Signals Batch file not found: {batch_file}")
            
            signal_batch = np.load(batch_file)
            self.cache.add_batch(f"signal_b_{batch_no}", signal_batch)
            
        if image_batch is None:
            batch_file = os.path.join(self.image_path, f"batch_{batch_no}.npy")
            
            if not os.path.exists(batch_file):
                raise FileNotFoundError(f"Image Batch file not found: {batch_file}")
            
            image_batch = np.load(batch_file)
            self.cache.add_batch(f"image_b_{batch_no}", image_batch)
        
        # Handle last batch edge case
        if sample_idx >= len(signal_batch) or sample_idx >= len(image_batch):
            raise IndexError(
                f"Sample index {sample_idx} out of bounds for batch {batch_no} "
                f"with {len(signal_batch)} samples"
            )
        # Get image and label
        signal = signal_batch[sample_idx].astype(np.float32)
        image = image_batch[sample_idx].astype(np.float32)
        label = self.labels[idx].astype(np.float32)
        
        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to tensors
        signal = torch.from_numpy(signal).float()
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        
        # Apply transforms if any
        if self.transform:
            signal = self.transform(signal)
            image = self.transform(image)
        
        return (signal, image), label


def create_fusion_dataloader(
    signal_path,
    image_path,
    labels_path,
    saved_batch_size=32,
    transform_type='gaf',
    batch_size=32,  # DataLoader batch size (can be different from saved_batch_size)
    shuffle=True,
    num_workers=4
):
    """
    Create dataloader for image data
    
    Args:
        image_path: Path to image directory
        labels_path: Path to labels file
        saved_batch_size: Size of saved batch files (default: 32)
        transform_type: 'gaf' or 'mtf'
        batch_size: DataLoader batch size (default: 16)
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    """
    dataset = ECGFusionDataset(
        image_path=image_path,
        signal_path= signal_path,
        labels_path=labels_path,
        saved_batch_size=saved_batch_size,
        transform_type=transform_type,
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

