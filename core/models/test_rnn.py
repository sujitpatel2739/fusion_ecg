"""
Test script for all RNN models
"""

import torch
from model import (
    create_gru, create_bigru, create_lstm, 
    create_bilstm, create_bigru_attention
)


def test_models():
    """Test all models with dummy data"""
    batch_size = 32
    num_leads = 3
    signal_length = 1000
    
    # Dummy input
    x = torch.randn(batch_size, num_leads, signal_length)
    
    # Create all models
    models = {
        'GRU': create_gru(),
        'BiGRU': create_bigru(),
        'LSTM': create_lstm(),
        'BiLSTM': create_bilstm(),
        'BiGRU-AdditiveAttention': create_bigru_attention(attention_type='additive'),
        'BiGRU-SelfAttention': create_bigru_attention(attention_type='self'),
    }
    
    print("="*70)
    print("TESTING RNN MODELS")
    print("="*70)
    print(f"Input shape: {x.shape}")
    print()
    
    for name, model in models.items():
        print(f"{name}:")
        print(f"  {'─'*60}")
        
        try:
            # Forward pass
            output = model(x)
            
            # Handle attention models (return tuple)
            if isinstance(output, tuple):
                logits, attention_weights = output
                print(f"  ✓ Logits shape: {logits.shape}")
                print(f"  ✓ Attention weights shape: {attention_weights.shape}")
            else:
                logits = output
                print(f"  ✓ Output shape: {logits.shape}")
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"  ✓ Total parameters: {num_params:,}")
            print(f"  ✓ Trainable parameters: {trainable_params:,}")
            print(f"  ✓ Model working correctly!")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
        
        print()
    
    print("="*70)
    print("✓ ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_models()