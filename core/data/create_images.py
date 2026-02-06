import os
import numpy as np
from pyts.image import GramianAngularField, MarkovTransitionField
from PIL import Image
from tqdm import tqdm

class ECGImageGenerator:
    def __init__(self, image_size=224, method='summation', batch_size=32):
        self.size = image_size
        self.gaf = GramianAngularField(image_size=self.size, method=method)
        self.mtf = MarkovTransitionField(image_size=self.size)
        self.batch_size = batch_size
    
    def normalize_image(self, image):
        """
        Normalize image from [-1, 1] or any range to [0, 255]
        """
        image_min = image.min()
        image_max = image.max()
        
        # Avoid division by zero
        if image_max - image_min == 0:
            return np.zeros_like(image, dtype=np.uint8)
        
        # Normalize to [0, 1] then scale to [0, 255]
        normalized = (image - image_min) / (image_max - image_min)
        return (normalized * 255).astype(np.uint8)
    
    def __call__(self, data, save_dir, partition='train'):
        """
        Generate and save GAF and MTF images
        
        Args:
            data: ECG signals (N, leads, signal_length)
            save_dir: Base directory to save images
            partition: 'train', 'val', or 'test'
        """
        # Create directories
        gaf_dir = os.path.join(save_dir, 'gaf')
        mtf_dir = os.path.join(save_dir, 'mtf')
        os.makedirs(gaf_dir, exist_ok=True)
        os.makedirs(mtf_dir, exist_ok=True)
        
        print(f"Generating images for {partition} set...")
        print(f"Total samples: {len(data)}")
        
        gaf_batch = []
        mtf_batch = []
        batch_idx = 0
        
        for sample_idx, signal in enumerate(tqdm(data, desc=f"Processing {partition}")):
            # signal shape: (leads, signal_length) = (3, 1000)
            
            gaf_leads = []
            mtf_leads = []
            
            # Process each lead
            for lead in signal:
                # lead shape: (signal_length,) = (1000,)
                
                # Generate GAF image
                gaf_image = self.gaf.fit_transform(lead.reshape(1, -1))[0]
                gaf_normalized = self.normalize_image(gaf_image)
                gaf_leads.append(gaf_normalized)
                
                # Generate MTF image
                mtf_image = self.mtf.fit_transform(lead.reshape(1, -1))[0]
                mtf_normalized = self.normalize_image(mtf_image)
                mtf_leads.append(mtf_normalized)
            
            # Stack leads to create multi-channel image
            # Shape: (3, 224, 224) - channels first
            gaf_stacked = np.stack(gaf_leads, axis=0)
            mtf_stacked = np.stack(mtf_leads, axis=0)
            
            gaf_batch.append(gaf_stacked)
            mtf_batch.append(mtf_stacked)
            
            # Save batch when it reaches batch_size
            if len(gaf_batch) == self.batch_size:
                # Convert to numpy array: (batch_size, 3, 224, 224)
                gaf_batch_array = np.array(gaf_batch)
                mtf_batch_array = np.array(mtf_batch)
                
                np.save(os.path.join(gaf_dir, f'batch_{batch_idx}.npy'), gaf_batch_array)
                np.save(os.path.join(mtf_dir, f'batch_{batch_idx}.npy'), mtf_batch_array)
                
                gaf_batch = []
                mtf_batch = []
                batch_idx += 1
        
        # Save remaining samples
        if len(gaf_batch) > 0:
            gaf_batch_array = np.array(gaf_batch)
            mtf_batch_array = np.array(mtf_batch)
            
            np.save(os.path.join(gaf_dir, f'batch_{batch_idx}.npy'), gaf_batch_array)
            np.save(os.path.join(mtf_dir, f'batch_{batch_idx}.npy'), mtf_batch_array)
        
        print(f"✓ Saved {batch_idx + 1} batches to {save_dir}")
        print(f"  - GAF images: {gaf_dir}")
        print(f"  - MTF images: {mtf_dir}")
        
        return batch_idx + 1 


# ============= MAIN EXECUTION ============================================================

X_TRAIN_SIGNAL_PATH = 'data/signals/X_train.npy'
X_TRAIN_IMAGES_PATH = 'data/images/train'
PARTITION = 'train'

# Load the train signals (No need to load labels)
print("Loading training data...")
X_train_signal = np.load(X_TRAIN_SIGNAL_PATH)
print(f"Loaded {X_train_signal.shape[0]} training signals with shape {X_train_signal.shape}")

# Create images from signals
image_generator = ECGImageGenerator(image_size=224, method='summation', batch_size=32)

# Generate and save images
num_batches = image_generator(
    data=X_train_signal, 
    save_dir=X_TRAIN_IMAGES_PATH,
    partition=PARTITION
)

print(f"\nImage generation complete!")
print(f"  Total batches: {num_batches}")
print(f"  Samples per batch: {image_generator.batch_size}")
print(f"  Last batch size: {X_train_signal.shape[0] % image_generator.batch_size or image_generator.batch_size}")