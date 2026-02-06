"""
DataLoader for ECG Images (GAF/MTF)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from precission_cache import PrecisionCache


class ECGImageDataset(Dataset):
    """
    Dataset for loading batched ECG images
    """
    def __init__(self, image_path, labels_path, saved_batch_size=32, transform_type='gaf', transform=None):
        """
        Args:
            image_path: Path to directory containing image batches (e.g., 'data/images/train/')
            labels_path: Path to labels .npy file
            saved_batch_size: Number of samples per saved batch file (default: 32)
            transform_type: 'gaf' or 'mtf'
            transform: Optional transforms to apply
        """
        self.image_path = os.path.join(image_path, transform_type)
        self.labels = np.load(labels_path)
        self.saved_batch_size = saved_batch_size
        self.transform_type = transform_type
        self.transform = transform
        self.cache = PrecisionCache(capacity=50)
        
        # Verify directory exists
        if not os.path.exists(self.image_path):
            raise ValueError(f"Image directory not found: {self.image_path}")
    
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
        image = batch[sample_idx].astype(np.float32)
        label = self.labels[idx].astype(np.float32)
        
        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to tensors
        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_image_dataloaders(
    image_path,
    labels_path,
    saved_batch_size=32,
    transform_type='gaf',
    batch_size=16,  # DataLoader batch size (can be different from saved_batch_size)
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
    dataset = ECGImageDataset(
        image_path=image_path,
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
        pin_memory=True
    )
    
    return loader


# ===================== Driver Code ===============================================================

if __name__ == "__main__":
    # Configuration
    TRAIN_IMAGE_PATH = 'data/images/train/'
    TRAIN_LABELS_PATH = 'data/labels/y_train.npy'
    VAL_IMAGE_PATH = 'data/images/validation/'
    VAL_LABELS_PATH = 'data/labels/y_validation.npy'
    TEST_IMAGE_PATH = 'data/images/test/'
    TEST_LABELS_PATH = 'data/labels/y_test.npy'
    
    SAVED_BATCH_SIZE = 32 
    DATALOADER_BATCH_SIZE = 16
    NUM_WORKERS = 4
    
    print("Creating dataloaders...")
    
    # Create dataloaders
    train_dataloader = create_image_dataloaders(
        image_path=TRAIN_IMAGE_PATH,
        labels_path=TRAIN_LABELS_PATH,
        saved_batch_size=SAVED_BATCH_SIZE,
        transform_type='gaf',
        batch_size=DATALOADER_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    
    val_dataloader = create_image_dataloaders(
        image_path=VAL_IMAGE_PATH,
        labels_path=VAL_LABELS_PATH,
        saved_batch_size=SAVED_BATCH_SIZE,
        transform_type='gaf',
        batch_size=DATALOADER_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    test_dataloader = create_image_dataloaders(
        image_path=TEST_IMAGE_PATH,
        labels_path=TEST_LABELS_PATH,
        saved_batch_size=SAVED_BATCH_SIZE,
        transform_type='gaf',
        batch_size=DATALOADER_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
