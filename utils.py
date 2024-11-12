"""
Adapted from https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/utils.py
Action format derived from VPT https://github.com/openai/Video-Pre-Training
"""

import torch
from einops import rearrange
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os


def sigmoid_beta_schedule(timesteps, start=-6, end=6, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule with adjusted parameters for more noise
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def cosine_beta_schedule_old(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    Modified to produce more typical alpha_cumprod values
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    
    # Cosine schedule
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    # Scale alphas_cumprod to typical range
    min_value = 0.0001  # You can adjust this value
    alphas_cumprod = alphas_cumprod * (1.0 - min_value) + min_value
    
    # Calculate betas
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    # Debug prints
    print(f"Alphas_cumprod range: [{alphas_cumprod.min():.6f}, {alphas_cumprod.max():.6f}]")
    print(f"Betas range: [{betas.min():.6f}, {betas.max():.6f}]")
    
    return torch.clip(betas, 0, 0.999)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Modified cosine schedule with even longer high-alpha period
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    
    # Modified cosine schedule with much higher power
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 8  # Increased power to 8
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    # Scale alphas_cumprod to typical range
    min_value = 0.0001
    alphas_cumprod = alphas_cumprod * (1.0 - min_value) + min_value
    
    # Calculate betas
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Linear schedule with more controlled noise levels
    """
    # Use smaller beta_end to prevent extreme noise at high timesteps
    beta_start = 1e-4
    beta_end = 0.01  # Reduced from 0.02
    
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Debug print
    print(f"Beta range: [{betas.min():.6f}, {betas.max():.6f}]")
    print(f"Alphas_cumprod range: [{alphas_cumprod.min():.6f}, {alphas_cumprod.max():.6f}]")
    
    return betas

@torch.inference_mode()
def visualize_step(self, x_curr, x_noisy, noise, v, pred=None, step=0, scaling_factor=0.07843137255, name:str = None):
    """Helper function to visualize intermediate steps"""
    # Print shape and value statistics for debugging
    print(f"\nDebug information for step {step}:")
    print(
        f"x_curr shape: {x_curr.shape}, range: [{x_curr.min():.3f}, {x_curr.max():.3f}]"
    )
    print(
        f"x_noisy shape: {x_noisy.shape}, range: [{x_noisy.min():.3f}, {x_noisy.max():.3f}]"
    )
    print(f"noise shape: {noise.shape}, range: [{noise.min():.3f}, {noise.max():.3f}]")

    # Get number of frames in sequence
    num_frames = x_curr.shape[1]

    # Create a figure with rows for each visualization type and columns for each frame
    fig, axes = plt.subplots(5, num_frames, figsize=(5 * num_frames, 25))

    # Helper function to convert latents to images
    def decode_latents(lat):
        lat = rearrange(lat, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            decoded = (self.vae.decode(lat / scaling_factor) + 1) / 2
            # Ensure values are in [0, 1]
            decoded = torch.clamp(decoded, 0, 1)
        return decoded.cpu()

    # 1. Original sequence
    orig_imgs = decode_latents(x_curr)
    orig_imgs = rearrange(orig_imgs, "(b t) c h w -> b t c h w", t=num_frames)

    # 2. Noisy sequence
    noisy_imgs = decode_latents(x_noisy)
    noisy_imgs = rearrange(noisy_imgs, "(b t) c h w -> b t c h w", t=num_frames)

    if pred is None:
        t = torch.full((1,), step, dtype=torch.long, device=x_curr.device)
        
        # Correct reconstruction formula
        x_start = (
            x_noisy - (1 - self.alphas_cumprod[t]).sqrt() * v
        ) / self.alphas_cumprod[t].sqrt()
        
        # Decode denoised images
        denoised_imgs = decode_latents(x_start)
        denoised_imgs = rearrange(denoised_imgs, "(b t) c h w -> b t c h w", t=num_frames)
    else:
        denoised_imgs = decode_latents(pred)
        denoised_imgs = rearrange(denoised_imgs, "(b t) c h w -> b t c h w", t=num_frames)


    # Visualize each frame
    for t in range(num_frames):
        # Original frame
        axes[0, t].imshow(make_grid(orig_imgs[:, t], nrow=1).permute(1, 2, 0))
        axes[0, t].set_title(
            f"Original Frame {t}\nRange: [{x_curr[0,t].min():.3f}, {x_curr[0,t].max():.3f}]"
        )
        axes[0, t].axis("off")

        # Noisy frame
        axes[1, t].imshow(make_grid(noisy_imgs[:, t], nrow=1).permute(1, 2, 0))
        axes[1, t].set_title(
            f"Noisy Frame {t}\nRange: [{x_noisy[0,t].min():.3f}, {x_noisy[0,t].max():.3f}]"
        )
        axes[1, t].axis("off")

        # Noise visualization (directly show the noise tensor)
 
        noise_vis = noise[:, t]  # Get noise for current timestep
        noise_grid = make_grid(noise_vis, nrow=1).permute(1, 2, 0).mean(-1).cpu()  # Average RGB channels after making grid
        im = axes[2, t].imshow(noise_grid, cmap="RdBu", interpolation="nearest")
        plt.colorbar(im, ax=axes[2, t])
        axes[2, t].set_title(
            f"Noise Frame {t}\nRange: [{noise_grid.min():.3f}, {noise_grid.max():.3f}]"
        )
        axes[2, t].axis('off')

        # Predicted noise visualization (row 3)
        v_vis = v[:, t]
        v_grid = make_grid(v_vis, nrow=1).permute(1, 2, 0).mean(-1).cpu()
        im = axes[3, t].imshow(v_grid, cmap='RdBu', interpolation='nearest')
        plt.colorbar(im, ax=axes[3, t])
        axes[3, t].set_title(
            f"Predicted Noise Frame {t}\nRange: [{v_grid.min():.3f}, {v_grid.max():.3f}]"
        )
        axes[3, t].axis('off')

        # Denoised frame (row 4)
        axes[4, t].imshow(make_grid(denoised_imgs[:, t], nrow=1).permute(1, 2, 0))
        axes[4, t].set_title(
            f"Denoised Frame {t}\nRange: [{denoised_imgs[0,t].min():.3f}, {denoised_imgs[0,t].max():.3f}]"
        )
        axes[4, t].axis('off')


    plt.suptitle(f"Step {step}", y=1.02, fontsize=16)
    plt.tight_layout()
    os.makedirs("debug_visualizations", exist_ok=True)
    save_path = f"debug_visualizations/sequence_step_{step}.png" if name is None else f"debug_visualizations/{name}"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
