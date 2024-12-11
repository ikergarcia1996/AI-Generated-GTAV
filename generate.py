"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""

import argparse

import torch
from accelerate import Accelerator
from einops import rearrange
from safetensors.torch import load_model
from torch import autocast
from torch.utils.data import DataLoader
from torchvision.io import write_video
from tqdm import tqdm

from model.dit import DiT_models
from model.vae import VAE_models
from train_dit import denoise_step
from utils import sigmoid_beta_schedule
from web_dataset import ImageDataset

torch.manual_seed(0)
torch.cuda.manual_seed(0)


@torch.inference_mode
def load_models(accelerator: Accelerator, dit_model_path: str, vae_model_path: str):
    # Load DiT model
    dit_model = DiT_models["DiT-S/2"]()
    missing_keys, unexpected_keys = load_model(dit_model, dit_model_path)
    if missing_keys or unexpected_keys:
        print(
            f"Error loading DiT model. Missing or unexpected keys. Please check the model.\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}"
        )

    # Load VAE model
    vae_model = VAE_models["vit-l-20-shallow-encoder"]()
    load_model(vae_model, vae_model_path)
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
        default="checkpoints/vit-l-20.safetensors",
        help="Path to VAE model checkpoint (default: checkpoints/vit-l-20-shallow-encoder.pt)",
    )

    parser.add_argument(
        "--noise_steps",
        type=int,
        default=100,
        help="Number of noise steps (default: 100)",
    )

    parser.add_argument(
        "--use_actions",
        action="store_true",
        help="Use actions (default: False). We will use W for all the frames.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="video1.mp4",
        help="Path to save the generated video (default: video1.mp4)",
    )

    args = parser.parse_args()

    assert torch.cuda.is_available()

    accelerator = Accelerator(
        mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16"
    )
    dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float16

    # Initialize models and parameters
    model, vae = load_models(accelerator, args.dit_model_path, args.vae_model_path)

    # Sampling parameters
    B = 1  # Batch size
    total_frames = args.total_frames
    n_prompt_frames = 4
    ddim_noise_steps = args.noise_steps
    noise_abs_max = 20
    stabilization_level = 15
    model.max_frames = 5
    print(
        f"We will generate {total_frames} frames, starting with {n_prompt_frames} frames."
    )
    print(f"Model max frames: {model.max_frames}")
    print(f"Noise steps: {ddim_noise_steps}")
    print(f"Stabilization level: {stabilization_level}")
    print(f"Noise absolute max: {noise_abs_max}")
    print(f"Actions is set to {args.use_actions}.")

    # Load input video
    test_dataset = ImageDataset(split="test", return_actions=args.use_actions)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
    test_loader = accelerator.prepare(test_loader)
    batch = next(iter(test_loader))
    video = batch["video"]
    actions = None if not args.use_actions else batch["actions"]

    if actions is not None:
        new_actions = torch.zeros(
            (
                actions.shape[0],
                total_frames - actions.shape[1],
                actions.shape[2],
            ),
            device=actions.device,
        )
        new_actions[:, :, 3] = 1  # Set W for all the new frames (drive straight)
        actions = torch.cat([actions, new_actions], dim=1)
    else:
        actions = None

    # Prepare input frames
    x = video[:, :n_prompt_frames]
    x = accelerator.prepare(x)
    x = vae_encode(x, vae, n_prompt_frames)

    max_noise_level = 1000
    ddim_noise_steps = args.noise_steps
    noise_range = torch.linspace(0, max_noise_level - 1, ddim_noise_steps + 1)
    betas = sigmoid_beta_schedule(max_noise_level).float().to(accelerator.device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    for i in tqdm(range(n_prompt_frames, total_frames)):
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=x.device)
        chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model.max_frames)
        x_old = x.clone()
        for noise_idx in reversed(range(0, ddim_noise_steps + 1)):
            x_pred, v_pred = denoise_step(
                dit_model=model,
                x_noisy=x,
                noise_idx=noise_idx,
                stabilization_level=stabilization_level,
                noise_range=noise_range,
                alphas_cumprod=alphas_cumprod,
                start_frame=start_frame,
                dtype=dtype,
                actions=actions,
            )

            # Update only the last frame
            x[:, -1:] = x_pred[:, -1:]
            """"
            if noise_idx == 0:
                visualize_step(
                    x_curr=x_old[:, start_frame:],
                    x_noisy=x[:, start_frame:],
                    noise=x[:, start_frame:],
                    v=v_pred,
                    step=noise_idx,
                    vae=vae,
                    alphas_cumprod=alphas_cumprod,
                    pred=x[:, start_frame:],
                    scaling_factor=0.07843137255,
                    name=f"frame_{i}",
                )
            """

    # Decode and save video
    x = rearrange(x, "b t c h w -> (b t) (h w) c")
    # print(x)
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
        x = (vae.decode(x / 0.07843137255) + 1) / 2
    x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)

    x = torch.clamp(x * 255, 0, 255).byte()
    # print(x)
    write_video(args.output_path, x[0].cpu(), fps=10)
    print(f"generation saved to {args.output_path}.")


if __name__ == "__main__":
    main()


# accelerate launch --mixed_precision bf16 generate.py --total-frames 128 --dit_model_path checkpoints/dit_continue_epoch_9_660000.safetensors --noise_steps 100 --output_path video_100_4.mp4
