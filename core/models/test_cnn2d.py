import torch
from cnn2d_alexnet import create_alexnet, create_alexnet_channel_attention, create_alexnet_cbam, create_alexnet_spatial_attention, create_alexnet_self_attention
from cnn2d_vggnet import VGGNet

def test_cnn2d_models():
    """Test all CNN2D variants"""
    batch_size = 8
    channels = 3
    height = 224
    width = 224
    
    # Dummy input (images)
    x = torch.randn(batch_size, channels, height, width)
    
    models = {
        'AlexNet (No Attention)': create_alexnet(),
        'AlexNet + Channel Attention': create_alexnet_channel_attention(),
        'AlexNet + Spatial Attention': create_alexnet_spatial_attention(),
        'AlexNet + CBAM': create_alexnet_cbam(),
        'AlexNet + Self-Attention': create_alexnet_self_attention(),
        'VGGNet': VGGNet(3, 128, 5, 0.4)
    }
    
    print("="*50)
    print("TESTING CNN2D MODELS")
    print("="*50)
    print(f"Input shape: {x.shape}\n")
    
    for name, model in models.items():
        print(f"{name}:")
        print(f"  {'-'*50}")
        
        try:
            # Forward pass
            output = model(x)
            print(f"  ✓ Output shape: {output.shape}")
            
            # Check output shape
            assert output.shape == (batch_size, 5), f"Wrong output shape: {output.shape}"
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"  ✓ Total parameters: {num_params:,}")
            print(f"  ✓ Trainable parameters: {trainable_params:,}")
            print(f"  ✓ Model working correctly!")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("="*50)
    print("✓ ALL TESTS COMPLETE")
    print("="*50)


if __name__ == "__main__":
    test_cnn2d_models()