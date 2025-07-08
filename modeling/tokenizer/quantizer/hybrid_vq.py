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

from typing import Mapping, Text, Tuple

import random
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast

from .quantizer import VectorQuantizer

class HybridVQ(VectorQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @autocast(enabled=False)
    def forward(self, z: torch.Tensor, skip_continuous_prob: float = 1.0) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z_quantized, result_dict = super(HybridVQ, self).forward(z)

        if self.use_l2_norm:
            z = F.normalize(z, p=2, dim=1)

        residual_features = z - z_quantized
        residual_features = rearrange(residual_features, 'b c h w -> b (h w) c')
        
        result_dict['residual_features'] = residual_features
        result_dict['z_quantized'] = z_quantized
        result_dict['features'] = z
        
        p = random.random()
        if p >= skip_continuous_prob:
            feature = z.clone()
        else:
            feature = z_quantized
        return feature, result_dict