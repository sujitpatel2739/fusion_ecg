"""
Model architecture definitions.
Includes GRU, BiGRU, BiLSTM, and attention-based variants.
Includes CNN2D-AlexNet, ResNet, VGGNet, and atention-based vatriants
"""

try:
    from .rnn import (
        create_gru,
        create_bigru,
        create_bilstm,
        create_bigru_attention,
    )
    from cnn2d_alexnet import (
        create_alexnet,
        create_alexnet_spatial_attention,
        create_alexnet_channel_attention,
        create_alexnet_cbam,
        create_alexnet_self_attention,
    )
    from cnn2d_resnet import ResNet18
    from cnn2d_vggnet import VGGNet
    __all__ = [
        'create_gru',
        'create_bigru',
        'create_bilstm',
        'create_bigru_attention',
        'create_alexnet',
        'create_alexnet_spatial_attention',
        'create_alexnet_channel_attention',
        'create_alexnet_cbam',
        'create_alexnet_self_attention',
        'ResNet18',
        'VGGNet',
    ]
except ImportError:
    # Handle case where model.py is not yet created
    __all__ = []
