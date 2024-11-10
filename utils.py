"""
Adapted from https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/utils.py
Action format derived from VPT https://github.com/openai/Video-Pre-Training
"""

import torch
from einops import rearrange
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
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


def visualize_step(self, x_curr, x_noisy, noise, step, scaling_factor=0.07843137255):
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
    fig, axes = plt.subplots(3, num_frames, figsize=(5 * num_frames, 15))

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
        if t < noise.shape[1]:
            noise_vis = noise[0, t].mean(0).cpu()  # Average across channels
            im = axes[2, t].imshow(noise_vis, cmap="RdBu", interpolation="nearest")
            plt.colorbar(im, ax=axes[2, t])
            axes[2, t].set_title(
                f"Noise Frame {t}\nRange: [{noise_vis.min():.3f}, {noise_vis.max():.3f}]"
            )
        else:
            axes[2, t].imshow(torch.zeros_like(noise[0, 0].mean(0).cpu()), cmap="RdBu")
            axes[2, t].set_title(f"No Noise (Frame {t})")
        axes[2, t].axis("off")

    plt.suptitle(f"Step {step}", y=1.02, fontsize=16)
    plt.tight_layout()
    os.makedirs("debug_visualizations", exist_ok=True)
    plt.savefig(f"debug_visualizations/sequence_step_{step}.png", bbox_inches="tight")
    plt.close()
