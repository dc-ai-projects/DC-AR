"""This files contains training loss implementation.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Ref:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
"""
from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast

class MLMLoss(torch.nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        self.label_smoothing = config.losses.label_smoothing
        self.loss_weight_unmasked_token = config.losses.loss_weight_unmasked_token
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing,
                                                   reduction="none")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                weights=None) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        inputs = rearrange(inputs, "b n c -> b c n")
        loss = self.criterion(inputs, targets)
        weights = weights.to(loss)
        loss_weights = (1.0 - weights) * self.loss_weight_unmasked_token + weights # set 0 to self.loss_weight_unasked_token
        loss = (loss * loss_weights).sum() / (loss_weights.sum() + 1e-8)
        # we only compute correct tokens on masked tokens
        correct_tokens = ((torch.argmax(inputs, dim=1) == targets) * weights).sum(dim=1) / (weights.sum(1) + 1e-8)
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}