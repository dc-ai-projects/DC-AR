from typing import Optional

from transformers import PretrainedConfig

class DCHTConfig(PretrainedConfig):
    model_type = 'dc_ht'

    def __init__(
        self,
        model_name: str = 'dc-ae-f32-proxy-in-1.0',
        pretrained_path: Optional[str] = None,

        codebook_size: int = 16384,
        codebook_embed_dim: int = 32,
        codebook_l2_norm: bool = True,
        codebook_show_usage: bool = True,
        commit_loss_beta: float = 0.25,
        entropy_loss_ratio: float = 0.0,
        quantizer_type: str = 'vq',

        disc_updated: bool = True,
        **kwargs
    ):
        super().__init__()

        self.model_name = model_name
        self.pretrained_path = pretrained_path

        self.codebook_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim
        self.codebook_l2_norm = codebook_l2_norm
        self.codebook_show_usage = codebook_show_usage
        self.commit_loss_beta = commit_loss_beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.quantizer_type = quantizer_type

        self.disc_updated = disc_updated
        
        