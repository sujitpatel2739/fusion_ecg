"""
DataLoader for ECG Images (GAF/MTF/MT or multiple)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from .precission_cache import PrecisionCache


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
            transform_type: 'gaf', 'mtf', or ['gaf', 'mtf'] to load both
            transform: Optional transforms to apply
        """
        self.labels = np.load(labels_path)
        self.saved_batch_size = saved_batch_size
        self.transform = transform
        self.cache = PrecisionCache(capacity=50)
        
        # Handle both string and list input for transform_type
        if isinstance(transform_type, str):
            self.transform_type = [transform_type]
        else:
            self.transform_type = transform_type
        
        # Build image paths for each transform type
        self.image_paths = []
        for t_type in self.transform_type:
            img_path = os.path.join(image_path, t_type)
            if not os.path.exists(img_path):
                raise ValueError(f"Image directory not found: {img_path}")
            self.image_paths.append(img_path)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Calculate which batch file and position within that batch
        batch_no = idx // self.saved_batch_size
        sample_idx = idx % self.saved_batch_size
        
        # Get image(s) from all transform types
        images = []
        for i, image_path in enumerate(self.image_paths):
            # Try to get from cache (use unique key combining path index and batch number)
            cache_key = (i, batch_no)
            batch = self.cache.get_batch(cache_key)
            
            # Load from disk if not in cache
            if batch is None:
                batch_file = os.path.join(image_path, f"batch_{batch_no}.npy")
                
                if not os.path.exists(batch_file):
                    raise FileNotFoundError(f"Batch file not found: {batch_file}")
                
                batch = np.load(batch_file)
                self.cache.add_batch(cache_key, batch)
            
            # Handle last batch edge case
            if sample_idx >= len(batch):
                raise IndexError(
                    f"Sample index {sample_idx} out of bounds for batch {batch_no} "
                    f"with {len(batch)} samples"
                )
            
            # Get image
            image = batch[sample_idx].astype(np.float32)
            
            # Normalize image to [0, 1] if needed
            if image.max() > 1.0:
                image = image / 255.0
            
            images.append(image)
        
        # Get label
        label = self.labels[idx].astype(np.float32)
        
        # Convert to tensors and apply transforms
        if len(images) > 1:
            # Return separate tensors for each image type
            image_tensors = []
            for image in images:
                img_tensor = torch.from_numpy(image).float()
                if self.transform:
                    img_tensor = self.transform(img_tensor)
                image_tensors.append(img_tensor)
            label = torch.from_numpy(label).float()
            return tuple(image_tensors), label
        else:
            # Return single image tensor
            image = torch.from_numpy(images[0]).float()
            label = torch.from_numpy(label).float()
            if self.transform:
                image = self.transform(image)
            return image, label


def create_image_dataloader(
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
        transform_type: 'gaf', 'mtf', or ['gaf', 'mtf'] to load both types
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
        pin_memory= torch.cuda.is_available()
    )
    
    return loader

