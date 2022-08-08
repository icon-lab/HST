import ml_collections

def get_small_config():
    """Returns Small HST configuration."""
    config = ml_collections.ConfigDict()
    config.d = 96
    config.num_blocks = [1, 1, 3, 1]
    config.num_attention_heads = [3, 6, 12, 24]
    config.pretrained_path = './model/imagenet_weights/hst_small_imagenet.pth'
    return config
  
def get_base_config():
    """Returns Base HST configuration."""
    config = ml_collections.ConfigDict()
    config.d = 96
    config.num_blocks = [1, 1, 9, 1]
    config.num_attention_heads = [3, 6, 12, 24]
    config.pretrained_path = './model/imagenet_weights/hst_base_imagenet.pth'
    return config
  
def get_large_config():
    """Returns Large HST configuration."""
    config = ml_collections.ConfigDict()
    config.d = 128
    config.num_blocks = [1, 1, 9, 1]
    config.num_attention_heads = [4, 8, 16, 32]
    config.pretrained_path = './model/imagenet_weights/hst_large_imagenet.pth'
    return config
  
