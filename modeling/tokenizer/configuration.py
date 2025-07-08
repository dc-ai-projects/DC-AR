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
        
        