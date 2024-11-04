"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""

import torch
from model.dit import DiT_models
from model.vae import VAE_models
from torchvision.io import read_video, write_video
from utils import one_hot_actions, sigmoid_beta_schedule
from tqdm import tqdm
from einops import rearrange
from torch import autocast


@torch.inference_mode
def main():
    assert torch.cuda.is_available()
    device = "cuda:0"

    # load DiT checkpoint
    # This model generates the video frames
    ckpt = torch.load("checkpoints/oasis500m.pt")
    model = DiT_models["DiT-S/2"]()
    model.load_state_dict(ckpt, strict=False)
    model = model.to(device).eval()

    # load VAE checkpoint
    # This model encodes and decodes the video frames
    vae_ckpt = torch.load("checkpoints/vit-l-20.pt")
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    vae.load_state_dict(vae_ckpt)
    vae = vae.to(device).eval()

    # sampling params
    B = 1  # Batch size
    total_frames = 32  # Number of frames to generate
    max_noise_level = 1000  # Maximum noise level for diffusion
    ddim_noise_steps = 100  # Number of denoising steps 
    noise_range = torch.linspace(
        -1, max_noise_level - 1, ddim_noise_steps + 1
    )  # Noise range (-1, 9, 19, ..., 999)
    noise_abs_max = 20  # Maximum absolute value for noise
    ctx_max_noise_idx = ddim_noise_steps // 10 * 3  # Context maximum noise index

    # get input video
    video_id = "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001"
    mp4_path = f"sample_data/{video_id}.mp4"
    actions_path = f"sample_data/{video_id}.actions.pt"
    video = (
        read_video(mp4_path, pts_unit="sec")[0].float() / 255
    )  # Normalize pixel values to [0, 1] range
    actions = one_hot_actions(torch.load(actions_path))
    offset = 100
    video = video[offset : offset + total_frames].unsqueeze(
        0
    )  # Selects a specific segment using offset. Size (1, 32, 256, 256, 3), Batch size, Time steps, Height, Width, Channels
    actions = actions[offset : offset + total_frames].unsqueeze(
        0
    )  # Selects a specific segment using offset, Size (1, 32, 25)

    # sampling inputs
    n_prompt_frames = 1
    x = video[:, :n_prompt_frames], #(1, 1, 360, 640, 3)
    x = x.to(device)
    actions = actions.to(device)

    # vae encoding
    # Encodes the input frames into the VAE's latent space.
    scaling_factor = 0.07843137255
    x = rearrange(x, "b t h w c -> (b t) c h w") # [1, 1, 360, 640, 3] -> [1, 3, 360, 640]
    H, W = x.shape[-2:]
    with torch.no_grad():
        x = vae.encode(x * 2 - 1).mean * scaling_factor # [1, 576, 16]
    x = rearrange(
        x,
        "(b t) (h w) c -> b t c h w",
        t=n_prompt_frames,
        h=H // vae.patch_size,
        w=W // vae.patch_size,
    ) # [1, 576, 16] -> [1, 1, 16, 18, 32]

    # Sets up the diffusion process parameters
    # Calculates alpha values used in the diffusion process
    betas = sigmoid_beta_schedule(max_noise_level).to(device) # [1000]
    alphas = 1.0 - betas # [1000]
    alphas_cumprod = torch.cumprod(alphas, dim=0) # [1000]
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1") # [1000] -> [1000, 1, 1, 1]

    # sampling loop. Generated frammes one by one.
    for i in tqdm(range(n_prompt_frames, total_frames)):
        # Generates random noise
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=device) # [1, 1, 16, 18, 32]
        chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max) # [1, 1, 16, 18, 32]
        x = torch.cat([x, chunk], dim=1) #  [1, 1, 16, 18, 32] + [1, 1, 16, 18, 32] -> [1, 2, 16, 18, 32]
        start_frame = max(0, i + 1 - model.max_frames) 

        for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
            # noise_idx: counts down from ddim_noise_steps (100) to 1, representing denoising steps
            
            ctx_noise_idx = min(noise_idx, ctx_max_noise_idx)
            # ctx_noise_idx: limits the noise level for context frames to ctx_max_noise_idx (30)
            
            t_ctx = torch.full((B, i), noise_range[ctx_noise_idx], dtype=torch.long, device=device)
            # t_ctx: creates tensor of size (batch_size, current_frame_index) filled with noise level for context frames
            
            t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
            # t: creates tensor of size (batch_size, 1) with current noise level for the frame being generated
            
            t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            # t_next: creates tensor with next noise level (one step less noisy)
            
            t_next = torch.where(t_next < 0, t, t_next)
            # Ensures t_next doesn't go below 0, using current noise level (t) if it would
            
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)
            # Combines context and current frame noise levels into single tensors

            # Sliding window implementation
            x_curr = x.clone()
            x_curr = x_curr[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]
            # Takes a window of recent frames to limit memory usage

            # Add noise to context frames
            ctx_noise = torch.randn_like(x_curr[:, :-1])
            ctx_noise = torch.clamp(ctx_noise, -noise_abs_max, +noise_abs_max)
            # Creates and clamps random noise for context frames
            
            x_curr[:, :-1] = (
                alphas_cumprod[t[:, :-1]].sqrt() * x_curr[:, :-1] +
                (1 - alphas_cumprod[t[:, :-1]]).sqrt() * ctx_noise
            )
            # Adds noise to context frames using alpha values from diffusion schedule

            # Model prediction
            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    v = model(x_curr, t, actions[:, start_frame : i + 1])
            # Uses DiT model to predict noise in half precision

            # Calculate original image estimate
            x_start = (
                alphas_cumprod[t].sqrt() * x_curr - 
                (1 - alphas_cumprod[t]).sqrt() * v
            )
            # Estimates the original, clean image using model's noise prediction

            # Calculate noise estimate
            x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (
                1 / alphas_cumprod[t] - 1
            ).sqrt()
            # Estimates the noise component

            # Predict next step
            x_pred = (
                alphas_cumprod[t_next].sqrt() * x_start +
                x_noise * (1 - alphas_cumprod[t_next]).sqrt()
            )
            # Combines clean image estimate and noise to predict next denoising step

            x[:, -1:] = x_pred[:, -1:]
            # Updates only the last frame being generated with the prediction

    # Decodes the generated latent representations back to pixel space
    x = rearrange(x, "b t c h w -> (b t) (h w) c")
    with torch.no_grad():
        x = (vae.decode(x / scaling_factor) + 1) / 2
    x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)

    # save video
    x = torch.clamp(x, 0, 1)
    x = (x * 255).byte()
    write_video("video.mp4", x[0].cpu(), fps=20)
    print("generation saved to video.mp4.")


if __name__ == "__main__":
    main()
