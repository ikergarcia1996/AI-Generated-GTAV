import os
from dataclasses import dataclass
from typing import Literal
import torch

import torch.nn as nn
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from einops import rearrange
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

import wandb
from hf_dataset import ImageDataset as HfDataset
from model.dit import DiT_models
from model.vae import VAE_models
from utils import sigmoid_beta_schedule
from web_dataset import ImageDataset as WebDataset
from dummy_dataset import ImageDataset as DummyDataset
from web_dataset import split_len
from torchvision.io import write_video
from utils import visualize_step
import logging

def checkerboard_loss(x):
    # Detect 2x2 checkerboard patterns
    checker_h = x[:, :, ::2, :] + x[:, :, 1::2, :]
    checker_w = x[:, :, :, ::2] + x[:, :, :, 1::2]
    return (checker_h**2).mean() + (checker_w**2).mean()

@torch.inference_mode()
def denoise_step(
    dit_model,
    x_noisy,
    actions,
    noise_idx,
    stabilization_level,
    noise_range,
    alphas_cumprod,
    start_frame=0,
    dtype=torch.bfloat16,
):
    """
    Performs a single denoising step in the diffusion process.

    Args:
        dit_model (nn.Module): The diffusion model
        x_noisy (torch.Tensor): The noisy input tensor
        actions (torch.Tensor): The actions tensor
        noise_idx (int): Current noise index in the denoising process
        stabilization_level (int): The stabilization level
        noise_range (torch.Tensor): The noise range
        alphas_cumprod (torch.Tensor): The cumulative product of alphas

    Returns:
        tuple: (x_pred, x_noisy_old, v_pred)
            - x_pred: The denoised prediction
            - x_noisy_old: The previous noisy state (for visualization)
            - v_pred: The predicted noise
    """
    # Create time tensors for context and target frames
    batch_size = x_noisy.shape[0]
    t_ctx = torch.full(
        (batch_size, x_noisy.shape[1] - 1),
        stabilization_level - 1,
        dtype=torch.long,
        device=x_noisy.device,
    )

    curr_noise = max(0, noise_range[noise_idx])  # Ensure non-negative
    next_noise = max(
        0, noise_range[max(0, noise_idx - 1)]
    )  # Ensure valid index and non-negative

    t = torch.full(
        (batch_size, 1),
        curr_noise,
        dtype=torch.long,
        device=x_noisy.device,
    )

    t_next = torch.full(
        (batch_size, 1),
        next_noise,
        dtype=torch.long,
        device=x_noisy.device,
    )
    t_next = torch.where(t_next < 0, t, t_next)

    # Concatenate time tensors
    t = torch.cat([t_ctx, t], dim=1)
    t_next = torch.cat([t_ctx, t_next], dim=1)

    # Apply sliding window
    x_curr = x_noisy.clone()
    x_curr = x_curr[:, start_frame:]
    t = t[:, start_frame:]
    t_next = t_next[:, start_frame:]
    if actions is not None:
        actions = actions[:, start_frame:]

    # Get model prediction
    with torch.autocast(
        "cuda",
        enabled=True,
        dtype=dtype,
    ):
        v_pred = dit_model(x_curr, t, actions)

    # Calculate denoising steps
    alpha_t = alphas_cumprod[t]
    x_start = alpha_t.sqrt() * x_curr - (1 - alpha_t).sqrt() * v_pred
    x_noise = ((1 / alpha_t).sqrt() * x_curr - x_start) / (
        (1 / alpha_t - 1).sqrt()
    )  # Add epsilon for numerical stability

    alpha_next = alphas_cumprod[t_next]
    alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])

    if noise_idx <= 0:  # Final step
        return x_start, v_pred

    # Compute prediction
    x_pred = alpha_next.sqrt() * x_start + (1 - alpha_next).sqrt() * x_noise

    return x_pred, v_pred


@dataclass
class TrainingConfig:
    vae_checkpoint: str = "checkpoints/vit-l-20.pt"
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    num_epochs: int = 5
    save_every: int = 2000
    gradient_accumulation_steps: int = 2
    seed: int = 42
    use_wandb: bool = True
    output_dir: str = "checkpoints"
    ddim_noise_steps: int = 16
    ddim_noise_steps_inference: int = 16
    ctx_max_noise_idx: int = 3  # (ddim_noise_steps // 10) * 3
    noise_abs_max: float = 20.0
    n_prompt_frames: int = 1
    min_learning_rate: float = 1e-6
    validation_batch_size: int = 8
    max_steps: int = -1  # -1 means no maximum steps limit
    validation_steps: int = 2000
    logging_steps: int = 5
    use_action_conditioning: bool = True
    warnup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    dataset_type: Literal["webdataset", "hfdataset", "dummy"] = "webdataset"
    pretrained_model: str = None
    model_name: str = "dit"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        data = cls(**config_dict)
        # Convert scientific notation strings to floats
        data.learning_rate = float(data.learning_rate)
        data.min_learning_rate = float(data.min_learning_rate)
        data.weight_decay = float(data.weight_decay)
        data.noise_abs_max = float(data.noise_abs_max)
        data.warnup_ratio = float(data.warnup_ratio)

        return data


class DiffusionTrainer:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb" if config.use_wandb else None,
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True)
            ],
        )

        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(project="diffusion-transformer", config=vars(config))

        # Set seed for reproducibility
        set_seed(config.seed)

        # Initialize models
        if config.pretrained_model is None:
            self.logger.info("Initializing new DiT model from scratch")
            self.dit = DiT_models["DiT-S/2"]()
        else:
            self.logger.info(
                f"Loading pretrained DiT model from {config.pretrained_model}"
            )
            self.dit = DiT_models["DiT-S/2"]()
            checkpoint = torch.load(config.pretrained_model, map_location="cpu")
            # Handle potential state_dict wrapper from accelerate
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            missing_keys, unexpected_keys = self.dit.load_state_dict(
                checkpoint, strict=False
            )
            if missing_keys:
                self.logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                self.logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        self.vae = VAE_models["vit-l-20-shallow-encoder"]()

        self.max_frames = self.dit.max_frames

        # Load VAE checkpoint and freeze
        vae_ckpt = torch.load(config.vae_checkpoint, map_location="cpu")
        self.vae.load_state_dict(vae_ckpt)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # Setup training
        self.optimizer = AdamW(
            self.dit.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-7,
        )

        # Calculate total steps for scheduler
        total_dataset_size = split_len("train")
        self.steps_per_epoch = total_dataset_size // (
            config.batch_size
            * self.accelerator.num_processes
            * config.gradient_accumulation_steps
        )
        self.total_training_steps = self.steps_per_epoch * config.num_epochs
        if config.max_steps > 0:
            self.total_training_steps = min(self.total_training_steps, config.max_steps)

        # Calculate warmup steps
        num_warmup_steps = int(self.config.warnup_ratio * self.total_training_steps)

        # Setup scheduler
        self.scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.total_training_steps,
            num_cycles=0.25,  # Standard cosine decay
            min_lr=self.config.min_learning_rate,
        )

        # Update prepare statement to include scheduler
        self.dit, self.vae, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.dit, self.vae, self.optimizer, self.scheduler
        )

        # Compile models for faster training (PyTorch 2.0+)
        """
        if torch.__version__ >= "2.0.0":
            #torch._dynamo.config.capture_scalar_outputs = True
            # Add compile configuratio

            self.dit = torch.compile(self.dit)
            self.vae = torch.compile(self.vae)
        else:
            self.logger.warning("PyTorch version < 2.0, skipping model compilation")
        """
        # Pre-compute and cache device tensors
        self.register_buffers()

    def register_buffers(self):
        """Pre-compute and cache tensors on device"""
        # Setup diffusion parameters

        self.max_noise_level = 1000
        self.ctx_max_noise_idx = self.config.ctx_max_noise_idx
        self.betas = sigmoid_beta_schedule(self.max_noise_level, clamp_min=0.000001).to(
            device=self.accelerator.device, dtype=torch.float32
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod = rearrange(self.alphas_cumprod, "T -> T 1 1 1")

        self.betas_inference = sigmoid_beta_schedule(self.max_noise_level, clamp_min=0.000001).to(
            device=self.accelerator.device, dtype=torch.float32
        )
        self.alphas_inference = 1.0 - self.betas_inference
        self.alphas_cumprod_inference = torch.cumprod(self.alphas_inference, dim=0)
        self.alphas_cumprod_inference = rearrange(self.alphas_cumprod_inference, "T -> T 1 1 1")


        # Update diffusion parameters with DDIM-style scheduling
        self.noise_range = (
            torch.linspace(
                0, self.max_noise_level - 1, self.config.ddim_noise_steps + 1
            )
            .long()
            .to(self.accelerator.device)
        )

        self.noise_range_inference = (
            torch.linspace(
                0, self.max_noise_level - 1, self.config.ddim_noise_steps_inference + 1
            )
            .long()
            .to(self.accelerator.device)
        )

        print(f"noise_range: {self.noise_range}")

        self.stabilization_level = 15

    @torch.inference_mode()
    def encode_frames(self, frames):
        """Encode frames using frozen VAE"""
        scaling_factor = 0.07843137255
        t = frames.shape[1]
        frames = rearrange(frames, "b t c h w -> (b t) c h w")

        with torch.no_grad():
            latents = self.vae.encode(frames * 2 - 1).mean * scaling_factor

        H, W = frames.shape[-2:]
        latents = rearrange(
            latents,
            "(b t) (h w) c -> b t c h w",
            t=t,
            h=H // self.vae.patch_size,
            w=W // self.vae.patch_size,
        )
        return latents

    @torch.inference_mode()
    def predict(self, test_loader, epoch, global_step, num_frames=16):
        """Generate a video from a prompt frame and optional actions"""
        self.dit.eval()

        # Move inputs to device and add batch dimension if needed
        prompt = next(iter(test_loader))
        prompt = prompt["video"]
        prompt = prompt[:1, : self.config.n_prompt_frames]  # Use only prompt frames
        if self.config.use_action_conditioning:
            actions = prompt["actions"][:1]
            if len(actions.shape[1]) < num_frames:
                new_actions = torch.zeros(
                    (
                        actions.shape[0],
                        num_frames - actions.shape[1],
                        actions.shape[2],
                    ),
                    device=actions.device,
                )
                new_actions[:, 3] = 1  # Set W for all the new frames (drive straight)
                actions = torch.cat([actions, new_actions], dim=1)

        else:
            actions = None

        # Encode prompt with VAE
        x = self.encode_frames(prompt)
        batch_size = x.shape[0]
        n_prompt_frames = x.shape[1]
        logging.info(
            f"\nStarting prediction with noise range: {self.noise_range.tolist()}"
        )
        # Generation loop
        logging.info(f"n_prompt_frames: {n_prompt_frames}.. num_frames: {num_frames}")
        for i in tqdm(
            range(n_prompt_frames, num_frames),
            desc="Generating test frames",
            disable=not self.accelerator.is_local_main_process,
        ):
            logging.info(f"\nGenerating frame {i}")

            new_frame = torch.randn(
                (batch_size, 1, *x.shape[2:]), device=self.accelerator.device
            )
            new_frame = torch.clamp(
                new_frame, -self.config.noise_abs_max, self.config.noise_abs_max
            )

            # Append the new noisy frame to existing context
            x = torch.cat([x, new_frame], dim=1)
            logging.info(
                f"Stabilized context frames range: [{x[:, :-1].min():.4f}, {x[:, :-1].max():.4f}]"
            )
            start_frame = max(0, i + 1 - self.dit.max_frames)
            # Progressive denoising of the last frame
            for noise_idx in reversed(range(0, self.config.ddim_noise_steps_inference + 1)):
                x_pred, v_pred = denoise_step(
                    dit_model=self.dit,
                    x_noisy=x,
                    actions=actions,
                    noise_idx=noise_idx,
                    stabilization_level=self.stabilization_level,
                    noise_range=self.noise_range_inference,
                    alphas_cumprod=self.alphas_cumprod_inference,
                    start_frame=start_frame,
                    dtype=torch.bfloat16
                    if self.accelerator.mixed_precision == "bf16"
                    else torch.float16,
                )
                # Update only the last frame
                x[:, -1:] = x_pred[:, -1:]

                # Visualize intermediate steps
                """
                visualize_step(
                    self,
                    x_curr=x_noisy[:1],
                    x_noisy=x_noisy[:1],
                    noise=torch.cat([ctx_noise, new_frame], axis=1)[:1],
                    v=v_pred[:1],
                    pred=torch.cat([x_noisy[:, :-1], x_start], dim=1)[:1],
                    step=global_step,
                    scaling_factor=0.07843137255,
                    name=f"{self.config.model_name}_gen_gs_{global_step}_frame_{i}_step_{step_idx}.png",
                )
                """

            logging.info(
                f"\nFinal frame {i} range: [{x[:, -1:].min():.4f}, {x[:, -1:].max():.4f}]"
            )

        # Decode latents to pixels
        scaling_factor = 0.07843137255
        x = rearrange(x, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            pixels = (self.vae.decode(x / scaling_factor) + 1) / 2
        pixels = rearrange(pixels, "(b t) c h w -> b t h w c", t=num_frames)

        # Convert to uint8 video
        pixels = torch.clamp(pixels * 255, 0, 255).byte()

        os.makedirs("debug_visualizations", exist_ok=True)
        video_path = f"debug_visualizations/test_{self.config.model_name}_{self.accelerator.process_index}_epoch_{epoch}_gs_{global_step}.mp4"
        write_video(
            video_path,
            pixels[0].cpu(),
            fps=10,
        )
        logging.info(f"generation saved to {video_path}.")

        self.dit.train()

    @torch.inference_mode()
    def predict_noise(self, test_loader, epoch, global_step):
        """Generate a video by adding noise to the last frame and then denoising it"""
        self.dit.eval()

        # Get input frames
        prompt = next(iter(test_loader))
        prompt = prompt["video"]
        prompt = prompt[:1]  # Take first batch only
        num_frames = prompt.shape[1]  # Get actual number of frames

        if self.config.use_action_conditioning:
            actions = prompt["actions"][:1]
        else:
            actions = None

        # Encode frames to latent space
        latents = self.encode_frames(prompt)
        batch_size = latents.shape[0]

        # Create a copy for noisy version
        x_noisy = latents.clone()

        # Add stabilized noise to context frames with the same noise level used in training
        ctx_noise = torch.randn_like(x_noisy[:, :-1])
        ctx_noise = torch.clamp(
            ctx_noise, -self.config.noise_abs_max, self.config.noise_abs_max
        )

        # Use stabilization level for context frames like in training
        t_ctx = torch.full(
            (batch_size, latents.shape[1] - 1),
            self.stabilization_level - 1,
            dtype=torch.long,
            device=self.accelerator.device,
        )
        alpha_ctx = self.alphas_cumprod[t_ctx]
        x_noisy[:, :-1] = (
            alpha_ctx.sqrt() * x_noisy[:, :-1] + (1 - alpha_ctx).sqrt() * ctx_noise
        )
        # Add noise to last frame using same noise schedule as training
        noise = torch.randn_like(x_noisy[:, -1:])
        actual_noise_level = self.noise_range[-1]
        t_last = torch.full(
            (batch_size, 1),
            actual_noise_level,
            dtype=torch.long,
            device=self.accelerator.device,
        )

        # Add initial noise
        alpha_t = self.alphas_cumprod[t_last]
        x_noisy[:, -1:] = (
            alpha_t.sqrt() * x_noisy[:, -1:] + (1 - alpha_t).sqrt() * noise
        )
        start_frame = max(0, num_frames - self.dit.max_frames)
        # Progressive denoising of the last frame only
        for noise_idx in reversed(range(0, self.config.ddim_noise_steps_inference + 1)):
            x_noisy_old = x_noisy.clone()
            x_pred, v_pred = denoise_step(
                dit_model=self.dit,
                x_noisy=x_noisy,
                actions=actions,
                noise_idx=noise_idx,
                stabilization_level=self.stabilization_level,
                noise_range=self.noise_range_inference,
                alphas_cumprod=self.alphas_cumprod_inference,
                start_frame=start_frame,
                dtype=torch.bfloat16
                if self.accelerator.mixed_precision == "bf16"
                else torch.float16,
            )
            # Update only the last frame
            x_noisy[:, -1:] = x_pred[:, -1:]
            # Visualize intermediate steps
            visualize_step(
                x_curr=latents[:1],
                x_noisy=x_noisy_old[:1],
                noise=torch.cat([ctx_noise, noise], dim=1)[
                    :1
                ],  # Make sure noise is properly shaped
                v=v_pred[:1],
                pred=x_pred[:1],  # Keep context frames unchanged
                step=global_step,
                scaling_factor=0.07843137255,
                name=f"{self.config.model_name}_noise_gs_{global_step}_pred_step_{noise_idx}.png",
                vae=self.vae,
                alphas_cumprod=self.alphas_cumprod,
            )

        self.dit.train()

    def _shared_step(
        self,
        frames,
        actions,
        global_step,
        visualize=False,
        is_training=True,
    ):
        """
        Shared logic between training and validation steps
        """
        batch_size = frames.shape[0]
        total_frames = frames.shape[1]

        # Encode frames to latent space
        latents = self.encode_frames(frames)
        total_loss = 0.0

        # Process frames sequentially after context frames
        for i in range(self.config.n_prompt_frames, total_frames):
            x_input = latents[:, : i + 1]
            if actions is not None:
                actions_input = actions[:, : i + 1]
            else:
                actions_input = None

            start_frame = max(0, i + 1 - self.max_frames)

            #training_progress = min(
            #    1.0, global_step / (self.total_training_steps * 0.5)
            #)  #  Use first half of training for curriculum
            #max_noise_idx = max(
            #    1, int(training_progress * self.config.ddim_noise_steps)
            #)  # Start with low noise (1) and gradually increase the maximum allowed noise level
            #print(f"max_noise_idx: {max_noise_idx}")
            # Random noise levels for target frame
            target_noise_idx = torch.randint(
                1,
                self.config.ddim_noise_steps + 1,
                (batch_size,),
                device=self.accelerator.device,
            )

            #print(target_noise_idx)

            ### For debugging, set max noise
            #target_noise_idx = torch.full(
            #    (batch_size,),
            #    self.config.ddim_noise_steps-5,
            #    device=self.accelerator.device,
            #)

            # Generate separate random noise indices for context frames
            ctx_noise_idx = torch.randint(
                1,
                self.config.ctx_max_noise_idx + 1,
                (batch_size,),
                device=self.accelerator.device,
            )

            ctx_noise_idx = torch.minimum(ctx_noise_idx, target_noise_idx)

            # Create time steps tensor and expand indices for broadcasting
            t = torch.zeros(
                (batch_size, total_frames),
                dtype=torch.long,
                device=self.accelerator.device,
            )

            # Set context frame noise levels (all frames except last)
            t[:, :-1] = (
                self.noise_range[ctx_noise_idx]
                .unsqueeze(1)
                .expand(-1, total_frames - 1)
            )

            # Set target frame noise levels (last frame)
            t[:, -1] = self.noise_range[target_noise_idx]

            # Apply sliding window
            x_curr = x_input[:, start_frame:]
            t = t[:, start_frame:]
            if actions_input is not None:
                actions_curr = actions_input[:, start_frame:]
            else:
                actions_curr = None

            # Add noise to context frames
            ctx_noise = torch.randn_like(x_curr[:, :-1])
            ctx_noise = torch.clamp(
                ctx_noise, -self.config.noise_abs_max, self.config.noise_abs_max
            )
            x_noisy = x_curr.clone()
            alpha_t = self.alphas_cumprod[t[:, :-1]]
            x_noisy[:, :-1] = (
                alpha_t.sqrt() * x_curr[:, :-1] + (1 - alpha_t).sqrt() * ctx_noise
            )

            # Add noise to current frame
            noise = torch.randn_like(x_curr[:, -1:])
            noise = torch.clamp(
                noise, -self.config.noise_abs_max, self.config.noise_abs_max
            )
            alpha_t = self.alphas_cumprod[t[:, -1:]]

            x_noisy[:, -1:] = (
                alpha_t.sqrt() * x_curr[:, -1:] + (1 - alpha_t).sqrt() * noise
            )
            # print(f"x_noise_range: [{x_noisy[:, -1:].min():.4f}, {x_noisy[:, -1:].max():.4f}]")
            v_target = alpha_t.sqrt() * noise - (1 - alpha_t).sqrt() * x_curr[:, -1:]
            # print(f"v_target_range: [{v_target[:, -1:].min():.4f}, {v_target[:, -1:].max():.4f}]")
            # Model prediction
            with self.accelerator.autocast():
                v_pred = self.dit(x_noisy, t, actions_curr)

                # 1. Simple noise prediction loss (L2)
                loss = nn.functional.mse_loss(v_pred[:, -1:], v_target)

                # 2. Reconstruction loss - predict original image
                #alpha_t = self.alphas_cumprod[t[:, -1:]]
                #x_start_pred = (
                #    alpha_t.sqrt() * x_noisy[:, -1:]
                #    - (1 - alpha_t).sqrt() * v_pred[:, -1:]
                #)
                # Clone x_curr to ensure it's not inference-only
                #x_curr_target = x_curr[:, -1:].clone()
                #recon_loss = nn.functional.mse_loss(x_start_pred, x_curr_target)
                #recon_loss_l1 = nn.functional.l1_loss(x_start_pred, x_curr_target)
                #print(
                #    f"recon_loss: {recon_loss}, recon_loss_l1: {recon_loss_l1}. Noise loss: {noise_loss}"
                #)
                # Combine losses with weights
                #loss = 0.2 * (recon_loss + recon_loss_l1) + noise_loss
                #tv_loss = checkerboard_loss(v_pred[:, -1:])
                #loss = noise_loss + 0.1 * tv_loss  # Adjust weight as needed

                #print(f"tv_loss: {tv_loss}, loss: {loss}")
            # print(f"v_pred_range: [{v_pred[:, -1:].min():.4f}, {v_pred[:, -1:].max():.4f}]")
            # print(f"loss: {loss}")
            if visualize:
                with torch.no_grad():
                    x_start = (
                        alpha_t.sqrt() * x_noisy[:, -1:]
                        - (1 - alpha_t).sqrt() * v_pred[:, -1:]
                     )
                    # print(f"x_start_range: [{x_start[:, -1:].min():.4f}, {x_start[:, -1:].max():.4f}]")
                    x_recon = torch.zeros_like(x_curr)
                    x_recon[:, :-1] = x_noisy[:, :-1]
                    x_recon[:, -1:] = x_start

                    visualize_step(
                        x_curr=x_curr[:1],
                        x_noisy=x_noisy[:1],
                        noise=torch.cat([ctx_noise, noise], axis=1)[:1],
                        v=v_pred[:1],
                        pred=x_recon[:1],
                        step=i,
                        scaling_factor=0.07843137255,
                        name=f"{self.config.model_name}_{'training' if is_training else 'validation'}_step_{global_step}.png",
                        vae=self.vae,
                        alphas_cumprod=self.alphas_cumprod,
                    )

            total_loss += loss

            if is_training:
                # Scale the loss and perform backward pass
                scaled_loss = loss / self.config.gradient_accumulation_steps
                self.accelerator.backward(scaled_loss)

        return total_loss / (total_frames - self.config.n_prompt_frames)

    def training_step(self, frames, actions, global_step, visualize: bool = False):
        """Single training step with context-aware noise scheduling"""
        if not hasattr(self, "_first_step_done"):
            rank = self.accelerator.process_index
            world_size = self.accelerator.num_processes
            logging.info(f"[GPU {rank}/{world_size}] Frames shape: {frames.shape}")
            logging.info(
                f"[GPU {rank}/{world_size}] Frame values - Min: {frames.min():.3f}, Max: {frames.max():.3f}, Mean: {frames.mean():.3f}"
            )
            if actions is not None:
                logging.info(
                    f"[GPU {rank}/{world_size}] Actions shape: {actions.shape}"
                )
                logging.info(
                    f"[GPU {rank}/{world_size}] Actions values - Min: {actions.min():.3f}, Max: {actions.max():.3f}, Mean: {actions.mean():.3f}"
                )
            self._first_step_done = True

        self.optimizer.zero_grad()
        return self._shared_step(
            frames, actions, global_step, visualize=visualize, is_training=True
        )

    @torch.inference_mode()
    def validation(self, val_loader, global_step):
        """Run validation loop"""
        self.dit.eval()
        val_losses = []
        total_dataset_size = split_len("validation")
        total_steps = total_dataset_size // (
            self.config.validation_batch_size * self.accelerator.num_processes
        )

        with tqdm(
            total=total_steps,
            desc="Validation",
            disable=not self.accelerator.is_local_main_process,
        ) as pbar:
            for batch in val_loader:
                frames = batch["video"]
                actions = (
                    batch["actions"] if self.config.use_action_conditioning else None
                )
                loss = self._shared_step(
                    frames,
                    actions,
                    global_step=global_step,
                    visualize=False,
                    is_training=False,
                )
                val_losses.append({"loss": loss.item()})
                pbar.update(1)

        self.dit.train()
        return val_losses

    def save_checkpoint(self, epoch, global_step):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            logging.info(
                f"Saving checkpoint at epoch {epoch+1} and step {global_step} to {self.config.output_dir}"
            )
            os.makedirs(self.config.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.config.output_dir,
                f"{self.config.model_name}_epoch_{epoch+1}_{global_step}.pt",
            )

            self.accelerator.save(
                self.accelerator.unwrap_model(self.dit).state_dict(), checkpoint_path
            )
            self.logger.warning(f"Saved checkpoint to {checkpoint_path}")

    def train(self, train_loader, val_loader):
        """Training loop"""
        # Prepare dataloader with accelerator
        train_loader, val_loader = self.accelerator.prepare(train_loader, val_loader)

        self.dit.train()
        total_dataset_size = split_len("train")
        steps_per_epoch = total_dataset_size // (
            self.config.batch_size
            * self.accelerator.num_processes
            * self.config.gradient_accumulation_steps
        )
        total_training_steps = steps_per_epoch * self.config.num_epochs
        if self.config.max_steps > 0:
            total_training_steps = min(total_training_steps, self.config.max_steps)

        with tqdm(
            total=total_training_steps,
            disable=not self.accelerator.is_local_main_process,
        ) as progress_bar:
            global_step = 0

            # Evaluate model before training
            """
            val_losses = self.validation(val_loader, global_step)
            avg_val_loss = sum(d["loss"] for d in val_losses) / len(val_losses)
            
            if self.accelerator.is_main_process and self.config.use_wandb:
                wandb.log(
                    {
                        "val_loss": avg_val_loss,
                        "epoch": 0,
                        "step": global_step,
                    }
                )
            
            self.predict(
                val_loader,
                epoch=0,
                global_step=global_step,
            )
            self.predict_noise(val_loader, epoch=0, global_step=global_step)
            """
            for epoch in range(self.config.num_epochs):
                accumulated_loss = 0.0  # Add accumulator
                for step, batch in enumerate(train_loader):
                    if (
                        self.config.max_steps > 0
                        and global_step >= self.config.max_steps
                    ):
                        print(f"Reached max steps: {self.config.max_steps}. Current step: {global_step}")
                        return
                    frames = batch["video"]
                    if self.config.use_action_conditioning:
                        actions = batch["actions"]
                    else:
                        actions = None

                    visualize = (
                        not hasattr(self, "_first_step_done")  # First step
                        or (
                            global_step > 0  # Past first step
                            and global_step % self.config.validation_steps
                            == 0  # Validation time
                            and (step + 1) % self.config.gradient_accumulation_steps
                            == 0  # Grad accumulation complete
                        )
                    )
                    loss = self.training_step(
                        frames, actions, global_step, visualize=visualize
                    )
                    accumulated_loss += loss  # Accumulate loss

                    # Update weights after gradient accumulation steps
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        # Calculate average loss over accumulation steps
                        avg_loss = (
                            accumulated_loss / self.config.gradient_accumulation_steps
                        )
                        accumulated_loss = 0.0  # Reset accumulator

                        self.accelerator.clip_grad_norm_(
                            self.dit.parameters(), self.config.max_grad_norm
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        progress_bar.update(1)
                        global_step += 1

                        if self.accelerator.is_main_process:
                            if (
                                self.config.use_wandb
                                and global_step % self.config.logging_steps == 0
                            ):
                                current_lr = self.scheduler.get_last_lr()[0]
                                wandb.log(
                                    {
                                        "train_loss": avg_loss,  # Use averaged loss
                                        "learning_rate": current_lr,
                                        "epoch": epoch,
                                        "step": global_step,
                                    }
                                )
                        if (
                            global_step > 0
                            and global_step % self.config.validation_steps == 0
                        ):
                            """
                            val_losses = self.validation(val_loader, global_step)
                            avg_val_loss = sum(d["loss"] for d in val_losses) / len(
                                val_losses
                            )
                            """

                            if (
                                self.accelerator.is_main_process
                                and self.config.use_wandb
                            ):
                                wandb.log(
                                    {
                                        "val_loss": avg_val_loss,
                                        "epoch": epoch,
                                        "step": global_step,
                                    }
                                )

                            self.predict(
                                val_loader,
                                epoch=0,
                                global_step=global_step,
                            )
                            self.predict_noise(
                                val_loader, epoch=0, global_step=global_step
                            )
                            

                        # Save checkpoint
                        if (
                            global_step > 0
                            and global_step % self.config.save_every == 0
                        ):
                            self.save_checkpoint(epoch, global_step)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config YAML file")
    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config)
    trainer = DiffusionTrainer(config)
    # Setup data loading

    if config.dataset_type == "webdataset":
        logging.info(
            "Using WebDataset. This will stream the dataset from the webdataset directory. Is memory efficient, but may be slow."
        )
        ImageDataset = WebDataset
    elif config.dataset_type == "hfdataset":
        logging.info(
            "Using HFDataset. This will load the dataset into memory. Is faster, but requires A LOT of RAM."
        )
        ImageDataset = HfDataset
    elif config.dataset_type == "dummy":
        logging.info("Using dummy dataset for testing purposes.")
        ImageDataset = DummyDataset
    else:
        raise ValueError(
            f"Invalid dataset type: {config.dataset_type}. Must be 'webdataset' or 'hfdataset'."
        )

    train_loader = DataLoader(
        ImageDataset(split="validation", return_actions=config.use_action_conditioning),
        batch_size=config.batch_size,
        num_workers=min(os.cpu_count(), 32),
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        ImageDataset(split="validation", return_actions=config.use_action_conditioning),
        batch_size=config.validation_batch_size,
        num_workers=min(os.cpu_count(), 8),
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )

    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
