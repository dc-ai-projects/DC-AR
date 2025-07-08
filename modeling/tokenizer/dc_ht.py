# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoConfig, AutoModel, PreTrainedModel

from .configuration import DCHTConfig
from .quantizer import VectorQuantizer as VQ
from .quantizer import HybridVQ
from .networks.dc_ae import Encoder, Decoder, DCAEConfig, dc_ae_f32, dc_ae_f16

class DCHT(PreTrainedModel):
    config_class = DCHTConfig

    def __init__(
        self,
        config: DCHTConfig
    ):
        super().__init__(config)

        if config.model_name in ["dc-ae-f32-in-1.0", "dc-ae-f32-mix-1.0"]:
            dc_ae_cfg = dc_ae_f32(config.model_name, config.codebook_embed_dim, config.pretrained_path)
        elif config.model_name in ['dc-ae-f16-in-1.0']:
            dc_ae_cfg = dc_ae_f16(config.model_name, config.codebook_embed_dim, config.pretrained_path)
        else:
            raise NotImplementedError
    
        self.cfg = dc_ae_cfg
        self.encoder = Encoder(dc_ae_cfg.encoder)
        self.decoder = Decoder(dc_ae_cfg.decoder)

        self.hybrid = False
        if config.quantizer_type == 'vq':
            self.quantize = VQ(config.codebook_size, config.codebook_embed_dim, 
                                config.commit_loss_beta, config.codebook_l2_norm)
        elif config.quantizer_type == 'hybrid_vq':
            self.quantize = HybridVQ(config.codebook_size, config.codebook_embed_dim,
                                    config.commit_loss_beta, config.codebook_l2_norm)
            self.hybrid = True

    def encode(self, x, **kwargs):
        h = self.encoder(x)
        quant, info = self.quantize(h, **kwargs)
        return quant, info

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def decode_tokens(self, tokens, residual_features = None):
        batch, seq_len = tokens.shape # B x N
        z_quantized = self.quantize.get_codebook_entry(
            tokens.reshape(-1)).reshape(batch, int(seq_len ** 0.5), int(seq_len ** 0.5), -1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        if self.hybrid and residual_features is not None:
            residual_features = residual_features.reshape(batch, int(seq_len ** 0.5), int(seq_len ** 0.5), -1)
            residual_features = rearrange(residual_features, 'b h w c -> b c h w').contiguous()
            decoded = self.decode(z_quantized + residual_features)
        else:
            decoded = self.decode(z_quantized)
        return decoded

    def forward(self, x, **kwargs):
        z_q, info = self.encode(x, **kwargs)
        decoded = self.decode(z_q)
        return decoded, info
    
AutoConfig.register("dc_ht", DCHTConfig)
AutoModel.register(DCHTConfig, DCHT)