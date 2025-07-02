import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath

from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf
from transformers import  AutoModel, AutoTokenizer

from modeling.utils import tokenize_fn
from modeling.modules.base_model import BaseModel
from modeling.diffusion import DiffLoss

from .sana.basic_modules import DWMlp, GLUMBConv, MBConvPreGLU, Mlp
from .sana.sana_blocks import (
    Attention,
    CaptionEmbedder,
    FlashAttention,
    LiteLA,
    MultiHeadCrossAttention,
    T2IFinalLayer,
)
from .sana.utils import auto_grad_checkpoint, to_2tuple



class DCARBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0,
        input_size=None,
        qk_norm=False,
        attn_type='flash',
        ffn_type='mlp',
        mlp_acts={"silu", "silu", None},
        linear_head_dim=32,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if attn_type == 'flash':
            self.attn = FlashAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=qk_norm,
                **block_kwargs,
            )
        elif attn_type == "linear":
            self_num_heads = hidden_size // linear_head_dim
            self.attn = LiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        elif attn_type == "vanilla":
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        else:
            raise ValueError(f"{attn_type} type is not defined.")

        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if ffn_type == "dwmlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = DWMlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        elif ffn_type == "glumbconv":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
            )
        elif ffn_type == "glumbconv_dilate":
            self.mlp = GLUMBConv(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=mlp_acts,
                dilation=2,
            )
        elif ffn_type == "mbconvpreglu":
            self.mlp = MBConvPreGLU(
                in_dim=hidden_size,
                out_dim=hidden_size,
                mid_dim=int(hidden_size * mlp_ratio),
                use_bias=(True, True, False),
                norm=None,
                act=("silu", "silu", None),
            )
        elif ffn_type == "mlp":
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
            )
        else:
            raise ValueError(f"{ffn_type} type is not defined.")
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, y, mask=None, attn_bias=None, **kwargs):
        B, N, C = x.shape

        x = x + self.drop_path(self.attn(self.norm1(x), attn_bias=attn_bias).reshape(B, N, C))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, pe_interpolation=1.0, base_size=16):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / pe_interpolation
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / pe_interpolation
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class DCAR(BaseModel, PyTorchModelHubMixin):
    def __init__(self, config):

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        self.target_codebook_size = config.model.vq_model.codebook_size
        self.image_seq_len = config.model.generator.image_seq_len
        self.mask_token_id = self.target_codebook_size
        self.hidden_size = config.model.generator.hidden_size
        self.num_hidden_layers = config.model.generator.num_hidden_layers
        self.num_attention_heads = config.model.generator.num_attention_heads
        self.intermediate_size = config.model.generator.intermediate_size

        self.text_tokenizer = AutoTokenizer.from_pretrained(config.model.text_model)
        text_encoder = AutoModel.from_pretrained(config.model.text_model)
        text_encoder.requires_grad_(False)
        text_encoder.eval().cuda()
        self.text_encoder = (text_encoder,)
        self.text_tokenizer_max_length = config.model.text_token_length

        self.reasoning = False
        if config.model.type == 'vq_dc_ae_reasoning':
            self.reasoning = True
            self.thinking_token_len = int(self.image_seq_len / (1 + config.model.vq_model.reasoning_downscale ** 2))
            self.generating_token_len = self.image_seq_len - self.thinking_token_len

        self.x_embedder = nn.Embedding(
            self.target_codebook_size + 1,
            self.hidden_size
        )
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_embedder = CaptionEmbedder(in_channels=config.model.context_dim, hidden_size=self.hidden_size, act_layer=approx_gelu, token_num=config.model.text_token_length)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.image_seq_len, self.hidden_size), requires_grad=False)
        self.pe_interpolation = config.model.generator.get('pe_interpolation', 1.0)
        self.base_size = config.model.generator.get('base_size', 16)
        drop_path = [x.item() for x in torch.linspace(0, config.model.generator.drop_path, self.num_hidden_layers)]

        self.blocks = nn.ModuleList([
            DCARBlock(self.hidden_size, self.num_attention_heads, self.intermediate_size / self.hidden_size,
                      drop_path=drop_path[i], input_size=(int(self.image_seq_len ** 0.5), int(self.image_seq_len ** 0.5)),
                      qk_norm=config.model.generator.get('qk_norm', False), attn_type=config.model.generator.get('attn_type', 'flash'), ffn_type=config.model.generator.get('ffn_type', 'mlp'),
                      mlp_acts=("silu", "silu", None), linear_head_dim=32,)
            for i in range(self.num_hidden_layers)
        ])
        self.final_layer = T2IFinalLayer(self.hidden_size, 1, self.target_codebook_size)

        diffusion_config = config.model.generator.diffusion
        self.diffloss = DiffLoss(
            target_channels=config.model.vq_model.token_size,
            z_channels=self.hidden_size,
            width=diffusion_config.width,
            depth=diffusion_config.depth,
            num_sampling_steps=diffusion_config.num_sampling_steps,
            sampler=diffusion_config.sampler,
            vae_scale=diffusion_config.vae_scale
        )
        self.diffusion_batch_mul = diffusion_config.batch_mul

        self.initialize_weights()

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        dict_config = OmegaConf.to_container(self.config)
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)
    
    def forward_diff_loss(self, z, target, mask=None):
        bs, seq_len, _ = target.shape
        target = target.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss
    

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if not self.reasoning:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.image_seq_len ** 0.5),
                pe_interpolation=self.pe_interpolation, base_size=self.base_size
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            generating_pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.generating_token_len ** 0.5),
                pe_interpolation=self.pe_interpolation, base_size=self.base_size
            )
            thinking_pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.thinking_token_len ** 0.5),
                pe_interpolation=self.pe_interpolation, base_size=self.base_size
            )
            pos_embed = np.concatenate([generating_pos_embed, thinking_pos_embed], axis=0)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.weight.data
        nn.init.trunc_normal_(w, mean=0.0, std=0.02)

        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    
    def encode_text(self, text):
        encoder_attention_mask = text != self.text_tokenizer.pad_token_id
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder[0].encoder(text).last_hidden_state

        return encoder_hidden_states, encoder_attention_mask


    def forward(self, input_ids=None, condition=None, encoder_hidden_states=None, encoder_attention_mask=None, residual_features=None, cond_drop_prob=0.1):
        if input_ids is None:
            raise NotImplementedError
        
        if not isinstance(condition, torch.Tensor):
            condition = tokenize_fn(
                condition,
                tokenizer=self.text_tokenizer,
                max_length=self.text_tokenizer_max_length,
                padding_mode='max_length',
            )
            condition = torch.stack(condition).cuda()
        

        x = self.x_embedder(input_ids) + self.pos_embed

        if encoder_hidden_states is None:
            with torch.no_grad():
                y, mask = self.encode_text(condition)
        else:
            y, mask = encoder_hidden_states, encoder_attention_mask

        y = self.y_embedder(y, self.training, cond_drop_prob)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1)
            y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.view(1, -1, x.shape[-1])

        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, y, y_lens)
        
        logits = self.final_layer(x)
        if residual_features is not None:
            diff_loss = self.forward_diff_loss(
                z=x, target=residual_features
            )
        else:
            diff_loss = None

        return logits, x, diff_loss

    def masking_input_tokens(self, input_tokens):
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device

        timesteps = torch.zeros((batch_size,), device=device).float().uniform_(0, 1.0)
        mask_ratio = torch.acos(timesteps) / (math.pi * 0.5) # arccos schedule
        mask_ratio = torch.clamp(mask_ratio, min=1e-6, max=1.)
        num_token_masked = (seq_len * mask_ratio).round().clamp(min=1)
        batch_randperm = torch.rand(batch_size, seq_len, device=device).argsort(dim=-1)
        masks = batch_randperm < rearrange(num_token_masked, 'b -> b 1')
        masked_tokens = torch.where(masks, self.mask_token_id, input_tokens)
        return masked_tokens, masks


    @torch.no_grad()
    def generate(self,
                 condition,
                 init_image_tokens=None,
                 init_residual_features=None,
                 init_mask=None,
                 guidance_scale=3.0,
                 guidance_decay="constant",
                 guidance_scale_pow=3.0,
                 randomize_temperature=4.5,
                 softmax_temperature_annealing=False,
                 num_sample_steps=8):
        if guidance_decay not in ["constant", "linear", "power-cosine"]:
            # contstant: constant guidance scale
            # linear: linear increasing the guidance scale as in MUSE
            # power-cosine: the guidance schedule from MDT
            raise ValueError(f"Unsupported guidance decay {guidance_decay}")

        if not isinstance(condition, torch.Tensor):
            condition = tokenize_fn(
                condition,
                tokenizer=self.text_tokenizer,
                max_length=self.text_tokenizer_max_length,
                padding_mode='max_length',
            )
            condition = torch.stack(condition).cuda()
            encoder_hidden_states, encoder_attention_mask = self.encode_text(condition)

        device = condition.device
        if init_image_tokens is not None:
            ids = init_image_tokens
        else:
            ids = torch.full((condition.shape[0], self.image_seq_len),
                              self.mask_token_id, device=device)

        cfg_scale = guidance_scale if guidance_decay == "constant" else 0.

        for step in range(num_sample_steps):
            ratio = 1. * (step + 1) / num_sample_steps
            annealed_temp = randomize_temperature * (1.0 - ratio)
            is_mask = (ids == self.mask_token_id)

            if guidance_decay == "power-cosine":
                guidance_scale_pow = torch.ones((1), device=device) * guidance_scale_pow
                scale_step = (1 - torch.cos(((step / num_sample_steps) ** guidance_scale_pow) * torch.pi)) * 1/2
                cfg_scale = (guidance_scale - 1) * scale_step + 1

            if cfg_scale != 0:
                logits, latents, _ = self.forward(
                    torch.cat([ids, ids], dim=0),
                    torch.cat([condition, condition], dim=0),
                    encoder_hidden_states=torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0),
                    encoder_attention_mask=torch.cat([encoder_attention_mask, encoder_attention_mask], dim=0),
                    cond_drop_prob=torch.cat([torch.zeros(ids.shape[0]), torch.ones(ids.shape[0])], dim=0).cuda()
                )
                cond_logits, uncond_logits = logits.chunk(2, dim=0)
                cond_latents, uncond_latents = latents.chunk(2, dim=0)
                if guidance_decay == "power-cosine":
                    logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
                else:
                    logits = cond_logits + (cond_logits - uncond_logits) * cfg_scale
            else:
                logits, latents, _ = self.forward(
                    ids, condition, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, cond_drop_prob=0.0
                )

            if softmax_temperature_annealing:
                softmax_temperature = 0.5 + 0.8 * (1 - ratio)
                logits = logits / softmax_temperature

            # Add gumbel noise
            def log(t, eps=1e-20):
                return torch.log(t.clamp(min=eps))
            def gumbel_noise(t):
                noise = torch.zeros_like(t).uniform_(0, 1)
                return -log(-log(noise))
            def add_gumbel_noise(t, temperature):
                return t + temperature * gumbel_noise(t)

            sampled_ids = add_gumbel_noise(logits, annealed_temp).argmax(dim=-1)
            sampled_logits = torch.squeeze(
                torch.gather(logits, dim=-1, index=torch.unsqueeze(sampled_ids, -1)), -1)
            sampled_ids = torch.where(is_mask, sampled_ids, ids)
            sampled_logits = torch.where(is_mask, sampled_logits, +np.inf).float()
            # masking
            mask_ratio = np.arccos(ratio) / (math.pi * 0.5)

            mask_len = torch.Tensor([np.floor(self.image_seq_len * mask_ratio)]).to(device)
            mask_len = torch.maximum(torch.Tensor([1]).to(device),
                                     torch.minimum(torch.sum(is_mask, dim=-1, keepdims=True) - 1,
                                                   mask_len))[0].squeeze()
            confidence = add_gumbel_noise(sampled_logits, annealed_temp)
            sorted_confidence, _ = torch.sort(confidence, axis=-1)
            cut_off = sorted_confidence[:, mask_len.long() - 1:mask_len.long()]
            masking = (confidence <= cut_off)
            if step == num_sample_steps - 1:
                ids = sampled_ids
                latents = torch.cat([cond_latents, uncond_latents], dim=0)
            else:
                ids = torch.where(masking, self.mask_token_id, sampled_ids)

            if guidance_decay == "linear":
                cfg_scale = ratio * guidance_scale

        bs, seq_len, _ = latents.shape
        predicted_latents = latents.reshape(bs * seq_len, -1)
        residual_features = self.diffloss.sample(
            z=predicted_latents, temperature=1.0, cfg=cfg_scale
        ).reshape(bs, seq_len, -1)
        residual_features, _ = residual_features.chunk(2, dim=0)

        if init_residual_features is not None:
            residual_features = torch.where(init_mask.unsqueeze(-1).expand_as(residual_features), init_residual_features, residual_features)

        return ids, residual_features