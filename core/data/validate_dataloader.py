from image_dataloader import train_dataloader, val_dataloader, test_dataloader

# ============= COMPREHENSIVE VALIDATION =============

def validate_dataloaders(train_loader, val_loader, test_loader):
    """Validate all dataloaders"""
    print("\n" + "="*60)
    print("DATALOADER VALIDATION")
    print("="*60)
    
    for name, loader in [('Train', train_loader), ('Val', val_loader), ('Test', test_loader)]:
        print(f"\n{name} Loader:")
        print(f"  Dataset size: {len(loader.dataset)}")
        print(f"  Number of batches: {len(loader)}")
        
        # Get first batch
        images, labels = next(iter(loader))
        print(f"  Batch image shape: {images.shape}")
        print(f"  Batch label shape: {labels.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Image dtype: {images.dtype}")
        
        # Verify all batches load
        total_samples = 0
        for batch_images, batch_labels in loader:
            total_samples += len(batch_images)
        
        print(f"  Total samples iterated: {total_samples}")
        print(f"  Expected: {len(loader.dataset)}")
        
        if total_samples == len(loader.dataset):
            print(f"  ✓ All samples accessible")
        else:
            print(f"  ✗ Missing {len(loader.dataset) - total_samples} samples!")
    
    print("\n" + "="*60)

# Run validation
validate_dataloaders(train_dataloader, val_dataloader, test_dataloader)