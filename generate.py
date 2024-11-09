"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""

import torch
from model.dit import DiT_models
from model.vae import VAE_models
from torchvision.io import write_video
from utils import sigmoid_beta_schedule
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from web_dataset import ImageDataset
from torch.utils.data import DataLoader

@torch.inference_mode
def load_models(device):
    # Load DiT model
    dit_ckpt = torch.load("checkpoints/oasis500m.pt")
    dit_model = DiT_models["DiT-S/2"]()
    dit_model.load_state_dict(dit_ckpt, strict=False)
    dit_model = dit_model.to(device).eval()

    # Load VAE model
    vae_ckpt = torch.load("checkpoints/vit-l-20.pt")
    vae_model = VAE_models["vit-l-20-shallow-encoder"]()
    vae_model.load_state_dict(vae_ckpt)
    vae_model = vae_model.to(device).eval()
    
    dit_model = torch.compile(dit_model)
    vae_model = torch.compile(vae_model)
    
    return dit_model, vae_model

@torch.inference_mode
def get_diffusion_params(device, max_noise_level=1000, ddim_steps=100):
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_steps + 1)
    betas = sigmoid_beta_schedule(max_noise_level).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    
    return noise_range, alphas_cumprod

@torch.inference_mode
def vae_encode(x, vae, n_prompt_frames, scaling_factor=0.07843137255):
    x = rearrange(x, "b t h w c -> (b t) c h w")
    H, W = x.shape[-2:]
    
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16 ):
        x = vae.encode(x * 2 - 1).mean * scaling_factor
        
    x = rearrange(
        x,
        "(b t) (h w) c -> b t c h w",
        t=n_prompt_frames,
        h=H // vae.patch_size,
        w=W // vae.patch_size,
    )
    return x

@torch.inference_mode
def generate_frame(model, x_curr, t, t_next, alphas_cumprod, noise_abs_max=20):
    # Add noise to context frames
    ctx_noise = torch.randn_like(x_curr[:, :-1])
    ctx_noise = torch.clamp(ctx_noise, -noise_abs_max, +noise_abs_max)
    
    x_curr[:, :-1] = (
        alphas_cumprod[t[:, :-1]].sqrt() * x_curr[:, :-1] +
        (1 - alphas_cumprod[t[:, :-1]]).sqrt() * ctx_noise
    )

    # Model prediction
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16 ):
        v = model(x_curr, t)

    # Calculate original image estimate
    x_start = (
        alphas_cumprod[t].sqrt() * x_curr - 
        (1 - alphas_cumprod[t]).sqrt() * v
    )

    # Calculate noise estimate
    x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (
        1 / alphas_cumprod[t] - 1
    ).sqrt()

    # Predict next step
    x_pred = (
        alphas_cumprod[t_next].sqrt() * x_start +
        x_noise * (1 - alphas_cumprod[t_next]).sqrt()
    )
    
    return x_pred

@torch.inference_mode
def main():
    assert torch.cuda.is_available()
    device = "cuda:0"

    # Initialize models and parameters
    model, vae = load_models(device)
    noise_range, alphas_cumprod = get_diffusion_params(device)
    
    # Sampling parameters
    B = 1  # Batch size
    total_frames = 32
    n_prompt_frames = 5
    ddim_noise_steps = 100
    noise_abs_max = 20
    ctx_max_noise_idx = ddim_noise_steps // 10 * 3

    # Load input video
    test_dataset = ImageDataset(split="test", return_actions=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
    video = next(iter(test_loader))["video"]
    
    # Prepare input frames
    x = video[:, :n_prompt_frames]
    x = x.to(device)
    x = vae_encode(x, vae, n_prompt_frames)

    # Generate frames
    for i in tqdm(range(n_prompt_frames, total_frames)):
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model.max_frames)

        for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
            ctx_noise_idx = min(noise_idx, ctx_max_noise_idx)
            
            t_ctx = torch.full((B, i), noise_range[ctx_noise_idx], dtype=torch.long, device=device)
            t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
            t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_next = torch.where(t_next < 0, t, t_next)
            
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            # Apply sliding window
            x_curr = x.clone()[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]

            x_pred = generate_frame(model, x_curr, t, t_next, alphas_cumprod, noise_abs_max)
            x[:, -1:] = x_pred[:, -1:]

    # Decode and save video
    x = rearrange(x, "b t c h w -> (b t) (h w) c")
    with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16 ):
        x = (vae.decode(x / 0.07843137255) + 1) / 2
    x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)
    
    x = torch.clamp(x * 255, 0, 255).byte()
    write_video("video.mp4", x[0].cpu(), fps=10)
    print("generation saved to video.mp4.")

if __name__ == "__main__":
    main()