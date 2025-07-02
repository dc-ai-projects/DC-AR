import argparse
import os
import random
import uuid

import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed
)

from modeling.tokenizer.dc_ht import DCHT
from utils.safety_check import is_dangerous
import utils.demo_util as demo_util

DESCRIPTION = (
    """# DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer"""
    + """\n<p>Note: We will replace unsafe prompts with a default prompt: \"A red heart.\"</p>"""
)
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo may not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "1") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "512"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_IMAGES_PER_PROMPT = 1


generator = None
tokenizer = None
safety_checker_tokenizer = None
safety_checker_model = None

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@spaces.GPU(enable_queue=True)
def generate(
    prompt: str,
    seed: int = 0,
    guidance_scale: float = 4.5,
    randomize_temperature: float = 1.5,
    randomize_seed: bool = False,
    num_sample_steps: int = 12,
    progress=gr.Progress(track_tqdm=True),
):
    global generator, tokenizer, safety_checker_tokenizer, safety_checker_model

    seed = int(randomize_seed_fn(seed, randomize_seed))

    if is_dangerous(safety_checker_tokenizer, safety_checker_model, prompt):
        prompt = "A red heart."

    generated_images = demo_util.sample_fn(
        generator=generator,
        tokenizer=tokenizer,
        conditions=[prompt],
        guidance_scale=guidance_scale,
        guidance_decay='constant',
        randomize_temperature=randomize_temperature,
        softmax_temperature_annealing=True,
        num_sample_steps=num_sample_steps,
        device=device,
        mean=0.5,
        std=0.5,
        return_tensor=True,
        hybrid=True,
        model_type='maskgit_t2i',
    ).clamp_(0., 1.)

    images = []
    sample_imgs_np = generated_images.clone().mul_(255).cpu().numpy()
    num_imgs = sample_imgs_np.shape[0]
    for img_idx in range(num_imgs):
        cur_img = sample_imgs_np[img_idx]
        cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
        cur_img_store = Image.fromarray(cur_img)
        images.append(cur_img_store)

    return images, seed


def main(args):

    global generator, tokenizer, safety_checker_tokenizer, safety_checker_model

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

    examples = [
        "melting apple",
        "neon holography crystal cat",
        "A dog that has been meditating all the time",
        "An astronaut riding a horse on the moon, oil painting by Van Gogh.",
        "8k uhd A man looks up at the starry sky, lonely and ethereal, Minimalism, Chaotic composition Op Art",
        "Full body shot, a French woman, Photography, French Streets background, backlighting, rim light, Fujifilm.",
        "Steampunk makeup, in the style of vray tracing, colorful impasto, uhd image, indonesian art, fine feather details with bright red and yellow and green and pink and orange colours, intricate patterns and details, dark cyan and amber makeup. Rich colourful plumes. Victorian style.",
    ]

    css = """
    .gradio-container{max-width: 560px !important}
    h1{text-align:center}
    """
    with gr.Blocks(css=css) as demo:
        gr.Markdown(DESCRIPTION)
        gr.DuplicateButton(
            value="Duplicate Space for private use",
            elem_id="duplicate-button",
            visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
        )
        with gr.Group():
            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run", scale=0)

            result = gr.Gallery(
                label="Result",
                columns=NUM_IMAGES_PER_PROMPT,
                show_label=False,
                # height=800,
            )
            with gr.Accordion("Advanced options", open=False):
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=args.seed,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=0.1,
                        maximum=20,
                        step=0.1,
                        value=4.5,
                    )
                with gr.Row():
                    randomize_temperature = gr.Slider(
                        label="Randomize Temperature",
                        minimum=0.1,
                        maximum=10,
                        step=0.1,
                        value=1.5,
                    )
                with gr.Row():
                    num_sample_steps = gr.Slider(
                        label="Number of Sample Steps",
                        minimum=8,
                        maximum=32,
                        step=1,
                        value=12,
                    )

        gr.Examples(
            examples=examples,
            inputs=prompt,
            outputs=[result, seed],
            fn=generate,
            cache_examples=CACHE_EXAMPLES,
        )

        gr.on(
            triggers=[
                prompt.submit,
                run_button.click,
            ],
            fn=generate,
            inputs=[
                prompt,
                seed,
                guidance_scale,
                randomize_temperature,
                randomize_seed,
                num_sample_steps,
            ],
            outputs=[result, seed],
            api_name="run",
        )

    demo.queue(max_size=20).launch(share=True)


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
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    main(args)