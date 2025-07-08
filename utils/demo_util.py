# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/bytedance/1d-tokenizer

import torch

from omegaconf import OmegaConf
from modeling.generator.dc_ar import DCAR

def get_config_cli():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf

def get_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf


def get_generator(config, tokenizer=None):
    if config.model.generator.model_type == 'DCAR':
        model_cls = DCAR
    else:
        raise ValueError(f"Unsupported model type {config.model.generator.model_type}")
    generator = model_cls(config)
    generator.load_state_dict(torch.load(config.experiment.generator_checkpoint, map_location="cpu"))
    generator.eval()
    generator.requires_grad_(False)
    return generator

@torch.no_grad()
def sample_fn(generator,
              tokenizer,
              conditions=None,
              guidance_scale=3.0,
              guidance_decay="constant",
              guidance_scale_pow=3.0,
              randomize_temperature=2.0,
              softmax_temperature_annealing=False,
              num_sample_steps=8,
              device="cuda",
              return_tensor=False,
              mean=0.0,
              std=1.0,
              hybrid=False,
              model_type='maskgit',
              init_image_tokens=None,
              init_residual_features=None,
              init_mask=None):
    generator.eval()
    tokenizer.eval()
    if model_type in ['maskgit']:
        if conditions is None:
        # goldfish, chicken, tiger, cat, hourglass, ship, dog, race car, airliner, teddy bear, random
            conditions = [1, 7, 282, 604, 724, 179, 751, 404, 850, torch.randint(0, 999, size=(1,))]
        if not isinstance(conditions, torch.Tensor):
            conditions = torch.LongTensor(conditions)
        conditions = conditions.to(device)
    elif model_type in ['maskgit_t2i', 'maskgit_t2i_inpainting']:
        # goldfish, chicken, tiger, cat, hourglass, ship, dog, race car, airliner, teddy bear, random
        if conditions is None:
            conditions = [
                "dog",
                "portrait photo of a girl, photograph, highly detailed face, depth of field",
                "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
                # MJV6 dataset
                "A young child is sitting on some pita bread next to some trees",
                "A slice of pizza",
                "A elegant sofa and chairs",
                "Group of people ski on the front of a field",
                "A person bent over on the mountain",
                "The action at home plate during a competitive baseball game as others watch",
                # overfitting prompts from training set
                "New Year card illustration, elegant nature style",
                "graphic for vintage stereo cassette, in the style of fauvist colors, kawaii aesthetic, flickr, color-streaked, holography, crisp outlines, strong use of color, white background, png",
                "russian snow day photos, art, in the style of adorable toy sculptures, orange and gold, cute cartoonish designs, neo-geo, frostpunk, emotive, orange",
                "Old vintage scrapbook wallpaper with line page, beautiful Black Eyed Susan pattern border, watercolor, high detail, HDR, self shadow, unique, intricate detail, hand-painted",
                "change to blue tang dynasty costumes, unchanged appearance",
            ]
    
    if model_type in ['maskgit_t2i', 'maskgit_t2i_inpainting']:
        generated_tokens = generator.generate(
            init_image_tokens=init_image_tokens,
            init_residual_features=init_residual_features,
            init_mask=init_mask,
            condition=conditions,
            guidance_scale=guidance_scale,
            guidance_decay=guidance_decay,
            guidance_scale_pow=guidance_scale_pow,
            randomize_temperature=randomize_temperature,
            softmax_temperature_annealing=softmax_temperature_annealing,
            num_sample_steps=num_sample_steps)
    else:
        raise NotImplementedError
    if isinstance(generated_tokens, tuple):
        generated_tokens, residual_features = generated_tokens
    if model_type in ['maskgit_t2i', 'maskgit_t2i_inpainting']:
        if hybrid:
            generated_image = tokenizer.decode_tokens(
                generated_tokens, residual_features
            ).mul_(std).add_(mean)
        else:
            generated_image = tokenizer.decode_tokens(
                generated_tokens, None
            ).mul_(std).add_(mean)
    else:
        raise NotImplementedError
    

    if return_tensor:
        return generated_image

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return generated_image
