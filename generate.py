"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""

import argparse

import torch
from accelerate import Accelerator
from einops import rearrange
from torch import autocast
from torch.utils.data import DataLoader
from torchvision.io import write_video
from tqdm import tqdm

from model.dit import DiT_models
from model.vae import VAE_models
from utils import sigmoid_beta_schedule
from web_dataset import ImageDataset
from generate_og import load_prompt

torch.manual_seed(0)
torch.cuda.manual_seed(0)


@torch.inference_mode
def load_models(accelerator: Accelerator, dit_model_path: str, vae_model_path: str):
    # Load DiT model
    dit_ckpt = torch.load(dit_model_path, weights_only=True)
    dit_ckpt = {
        k.replace("_orig_mod.", ""): v for k, v in dit_ckpt.items()
    }  # Remove torch.compile prefix
    dit_model = DiT_models["DiT-S/2"]()
    missing_keys, unexpected_keys = dit_model.load_state_dict(dit_ckpt, strict=False)
    if missing_keys or unexpected_keys:
        print(
            f"Error loading DiT model. Missing or unexpected keys. Please check the model.\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}"
        )

    # Load VAE model
    vae_ckpt = torch.load(vae_model_path, weights_only=True)
    vae_model = VAE_models["vit-l-20-shallow-encoder"]()
    vae_model.load_state_dict(vae_ckpt)

    dit_model, vae_model = accelerator.prepare(dit_model, vae_model)
    # dit_model = torch.compile(dit_model)
    # vae_model = torch.compile(vae_model)

    return dit_model, vae_model


@torch.inference_mode
def vae_encode(x, vae, n_prompt_frames, scaling_factor=0.07843137255):
    x = rearrange(x, "b t c h w -> (b t) c h w")
    H, W = x.shape[-2:]

    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        x = vae.encode(x * 2 - 1).mean * scaling_factor

    x = rearrange(
        x,
        "(b t) (h w) c -> b t c h w",
        t=n_prompt_frames,
        h=H // vae.patch_size,
        w=W // vae.patch_size,
    )
    # print(x)
    return x


@torch.inference_mode
def main():
    parser = argparse.ArgumentParser(description="Video generation script")
    parser.add_argument(
        "--total-frames",
        type=int,
        default=32,
        help="Total number of frames to generate (default: 32)",
    )

    parser.add_argument(
        "--dit_model_path",
        type=str,
        default="checkpoints/oasis500m.pt",
        help="Path to DiT model checkpoint (default: checkpoints/oasis500m.pt)",
    )

    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="checkpoints/vit-l-20.pt",
        help="Path to VAE model checkpoint (default: checkpoints/vit-l-20-shallow-encoder.pt)",
    )

    parser.add_argument(
        "--noise_steps",
        type=int,
        default=100,
        help="Number of noise steps (default: 100)",
    )

    args = parser.parse_args()

    assert torch.cuda.is_available()

    accelerator = Accelerator(
        mixed_precision="bf16",
    )

    # Initialize models and parameters
    model, vae = load_models(accelerator, args.dit_model_path, args.vae_model_path)

    # Sampling parameters
    B = 1  # Batch size
    total_frames = args.total_frames
    n_prompt_frames = 1
    ddim_noise_steps = args.noise_steps
    noise_abs_max = 20
    stabilization_level = 15

    # Load input video
    test_dataset = ImageDataset(split="test", return_actions=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
    test_loader = accelerator.prepare(test_loader)
    video = next(iter(test_loader))["video"]
    video = load_prompt("sample_data/gtaV1.jpg").to(accelerator.device)
    # Prepare input frames
    x = video[:, :n_prompt_frames]
    x = accelerator.prepare(x)
    x = vae_encode(x, vae, n_prompt_frames)

    max_noise_level = 1000
    ddim_noise_steps = args.noise_steps
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
    noise_abs_max = 20
    stabilization_level = 15
    betas = sigmoid_beta_schedule(max_noise_level).float().to(accelerator.device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    for i in tqdm(range(n_prompt_frames, total_frames)):
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=x.device)
        chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model.max_frames)

        for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
            t_ctx = torch.full(
                (B, x.shape[1] - 1),
                stabilization_level - 1,
                dtype=torch.long,
                device=accelerator.device,
            )
            t = torch.full(
                (B, 1),
                noise_range[noise_idx],
                dtype=torch.long,
                device=accelerator.device,
            )

            t_next = torch.full(
                (B, 1),
                noise_range[noise_idx - 1],
                dtype=torch.long,
                device=accelerator.device,
            )
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            x_curr = x.clone()
            x_curr = x_curr[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]

            with autocast("cuda", dtype=torch.bfloat16):
                v_pred = model(x_curr, t)  # ,  actions[:, start_frame : i + 1])

            alpha_t = alphas_cumprod[t]
            x_start = alpha_t.sqrt() * x_curr - (1 - alpha_t).sqrt() * v_pred
            x_noise = ((1 / alpha_t).sqrt() * x_curr - x_start) / (
                1 / alpha_t - 1
            ).sqrt()

            alpha_next = alphas_cumprod[t_next]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])

            if noise_idx == 1:  # Final step
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])

            # Compute prediction
            x_pred = alpha_next.sqrt() * x_start + (1 - alpha_next).sqrt() * x_noise

            # Update only the last frame
            x[:, -1:] = x_pred[:, -1:]

    # Decode and save video
    x = rearrange(x, "b t c h w -> (b t) (h w) c")
    # print(x)
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        x = (vae.decode(x / 0.07843137255) + 1) / 2
    x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)

    x = torch.clamp(x * 255, 0, 255).byte()
    # print(x)
    write_video("video1.mp4", x[0].cpu(), fps=20)
    print("generation saved to video1.mp4.")


if __name__ == "__main__":
    main()
