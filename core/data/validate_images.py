import numpy as np
import matplotlib.pyplot as plt

# ============= VALIDATION CODE ==============================================
print("\nValidating generated images...")

# Load first batch
gaf_batch_0 = np.load('data/images/train/gaf/batch_0.npy')
mtf_batch_0 = np.load('data/images/train/mtf/batch_0.npy')

print(f"GAF batch shape: {gaf_batch_0.shape}")  # Should be (32, 3, 224, 224)
print(f"MTF batch shape: {mtf_batch_0.shape}")  # Should be (32, 3, 224, 224)

# Visualize first sample

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# GAF - 3 leads
for i in range(3):
    axes[0, i].imshow(gaf_batch_0[0, i], cmap='viridis')
    axes[0, i].set_title(f'GAF - Lead {i}')
    axes[0, i].axis('off')

# MTF - 3 leads
for i in range(3):
    axes[1, i].imshow(mtf_batch_0[0, i], cmap='viridis')
    axes[1, i].set_title(f'MTF - Lead {i}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('sample_images_validation.png', dpi=150)
plt.close()

print("✓ Validation complete! Check sample_images_validation.png")