import argparse
import os
import json
import random
from typing import Optional, Tuple
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision
from PIL import Image
import numpy as np

import utils.demo_util as demo_util
from utils import default_prompts
from utils.safety_check import is_dangerous

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from modeling.tokenizer.dc_ht import DCHT


def save_images(sample_imgs, sample_folder_dir, store_separately, prompts):
    if not store_separately and len(sample_imgs) > 1:
        grid = torchvision.utils.make_grid(sample_imgs, nrow=12)
        grid_np = grid.to(torch.float16).permute(1, 2, 0).mul_(255).cpu().numpy()

        os.makedirs(sample_folder_dir, exist_ok=True)
        grid_np = Image.fromarray(grid_np.astype(np.uint8))
        grid_np.save(os.path.join(sample_folder_dir, f"sample_images.png"))
        print(f"Example images are saved to {sample_folder_dir}")
    else:
        # bs, 3, r, r
        sample_imgs_np = sample_imgs.mul_(255).cpu().numpy()
        num_imgs = sample_imgs_np.shape[0]
        os.makedirs(sample_folder_dir, exist_ok=True)
        for img_idx in range(num_imgs):
            cur_img = sample_imgs_np[img_idx]
            cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
            cur_img_store = Image.fromarray(cur_img)
            cur_img_store.save(os.path.join(sample_folder_dir, f"{img_idx:06d}.png"))
            print(f"Image {img_idx} saved.")

    with open(os.path.join(sample_folder_dir, "prompt.txt"), "w") as f:
        f.write("\n".join(prompts))


def main(args):
    device = torch.device("cuda")

    torch.manual_seed(args.seed)
    
    config = demo_util.get_config(args.config)
    tokenizer = AutoModel.from_pretrained(config.experiment.tokenizer_checkpoint)
    tokenizer.eval()
    tokenizer.requires_grad_(False)

    generator = demo_util.get_generator(config)

    tokenizer = tokenizer.to(device)
    generator = generator.to(device)

    safety_checker_tokenizer = AutoTokenizer.from_pretrained(args.shield_model_path)
    safety_checker_model = AutoModelForCausalLM.from_pretrained(
        args.shield_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).to(device)

    prompts = []
    if args.prompt_list:
        prompts = args.prompt_list
    elif args.prompt:
        prompts = [args.prompt]
    else:
        print(
            "No prompt is provided. Will randomly sample 4 prompts from default prompts."
        )
        prompts = random.sample(default_prompts, 4)

    for idx, prompt in enumerate(prompts):
        if is_dangerous(
            safety_checker_tokenizer, safety_checker_model, prompt
        ):
            prompts[idx] = random.sample(default_prompts, 1)[0]
            print(
                f"Detected Unsafe prompt with index {idx}, will replace by one of default prompts."
            )
    
    start_time = time.time()
    generated_images = demo_util.sample_fn(
        generator=generator,
        tokenizer=tokenizer,
        conditions=prompts,
        randomize_temperature=config.model.generator.randomize_temperature,
        softmax_temperature_annealing=True,
        num_sample_steps=config.model.generator.get('num_steps', 256),
        guidance_scale=config.model.generator.guidance_scale,
        guidance_decay=config.model.generator.get('guidance_decay', 'power-cosine'),
        guidance_scale_pow=config.model.generator.get('guidance_scale_pow', 2.75),
        return_tensor=True,
        mean=config.dataset.eval.mean,
        std=config.dataset.eval.std,
        device=device,
        hybrid=config.model.get('hybrid', False),
        model_type=config.model.generator.get('type', 'maskgit')
    ).clamp_(0., 1.)

    total_time = time.time() - start_time
    print(f"Generate {len(prompts)} images take {total_time:2f}s.")

    save_images(
        generated_images.clone(), 
        args.sample_folder_dir,
        args.store_seperately,
        prompts
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference/dc_ar_t2i_512.yaml",
    )
    parser.add_argument(
        "--shield_model_path",
        type=str,
        help="The path to shield model, we employ ShieldGemma-2B by default.",
        default="pretrained_models/shieldgemma-2b",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="A single prompt.",
        default=""
    )
    parser.add_argument(
        "--prompt_list",
        type=list[str],
        default=[]
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--sample_folder_dir",
        type=str,
        help="The folder where the image samples are stored",
        default="samples/examples/",
    )
    parser.add_argument(
        "--store_seperately",
        help="Store image samples in a grid or separately, set to False by default.",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)