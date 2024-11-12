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
from utils import sigmoid_beta_schedule, cosine_beta_schedule, linear_beta_schedule
from web_dataset import ImageDataset as WebDataset
from dummy_dataset import ImageDataset as DummyDataset
from web_dataset import split_len
from torchvision.io import write_video
from utils import visualize_step


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
        total_training_steps = self.steps_per_epoch * config.num_epochs
        if config.max_steps > 0:
            total_training_steps = min(total_training_steps, config.max_steps)

        # Calculate warmup steps
        num_warmup_steps = int(self.config.warnup_ratio * total_training_steps)

        # Setup scheduler
        self.scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
            num_cycles=0.25,  # Standard cosine decay
            min_lr=self.config.min_learning_rate,
        )

        # Update prepare statement to include scheduler
        self.dit, self.vae, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.dit, self.vae, self.optimizer, self.scheduler
        )

        # Compile models for faster training (PyTorch 2.0+)
        # self.dit = torch.compile(self.dit)
        # self.vae = torch.compile(self.vae)

        # Pre-compute and cache device tensors
        self.register_buffers()

    def register_buffers(self):
        """Pre-compute and cache tensors on device"""
        # Setup diffusion parameters
        self.max_noise_level = 1000
        self.betas = cosine_beta_schedule(self.max_noise_level).to(
            self.accelerator.device
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod = rearrange(self.alphas_cumprod, "T -> T 1 1 1")

        # Update diffusion parameters with DDIM-style scheduling
        self.noise_range = (
            torch.linspace(
                -1, self.max_noise_level - 1, self.config.ddim_noise_steps + 1
            )
            .long()
            .to(self.accelerator.device)
        )

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
    def validation(self, val_loader):
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
                if self.config.use_action_conditioning:
                    actions = batch["actions"]
                else:
                    actions = None

                batch_size = frames.shape[0]
                total_frames = frames.shape[1]
                total_loss = 0.0

                # Similar logic to training_step but without gradient computation
                latents = self.encode_frames(frames)

                for i in range(self.config.n_prompt_frames, total_frames):
                    x_input = latents[:, : i + 1]
                    if actions is not None:
                        actions_input = actions[:, : i + 1]
                    else:
                        actions_input = None
                    start_frame = max(
                        0, i + 1 - self.max_frames
                    )  # Always 0 in our data

                    t = torch.randint(
                        0,
                        self.noise_range[-1],
                        (batch_size, total_frames),
                        device=self.accelerator.device,
                    )

                    # Prepare noise levels for context and current frame
                    t = torch.full(
                        (batch_size, i + 1),  # Shape matches all frames
                        self.stabilization_level
                        - 1,  # Default to context noise level
                        dtype=torch.long,
                        device=self.accelerator.device,
                    )
                    t[:, :-1] = self.stabilization_level - 1

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
                    x_noisy[:, :-1] = (
                        self.alphas_cumprod[t[:, :-1]].sqrt() * x_noisy[:, :-1]
                        + (1 - self.alphas_cumprod[t[:, :-1]]).sqrt() * ctx_noise
                    )

                    # Add noise to current frame
                    noise = torch.randn_like(x_curr[:, -1:])
                    noise = torch.clamp(
                        noise, -self.config.noise_abs_max, self.config.noise_abs_max
                    )
                    x_noisy[:, -1:] = (
                        self.alphas_cumprod[t[:, -1:]].sqrt() * x_curr[:, -1:]
                        + (1 - self.alphas_cumprod[t[:, -1:]]).sqrt() * noise
                    )
                    # Model prediction
                    with torch.autocast(
                        "cuda",
                        enabled=True,
                        dtype=torch.bfloat16 if self.accelerator.mixed_precision == "bf16" else torch.float16,
                    ):
                        v = self.dit(x_noisy, t, actions_curr)

                    # The model predicts v (noise), and we can directly compare it with the noise we added
                    loss = nn.functional.mse_loss(v[:, -1:], noise) 

                    total_loss += loss

                avg_loss = total_loss / (total_frames - self.config.n_prompt_frames)
                val_losses.append({"loss": avg_loss.item()})
                pbar.update(1)

        self.dit.train()
        return val_losses
    
    @torch.inference_mode()
    def predict(self, test_loader, epoch, global_step, num_frames=5):
        """Generate a video from a prompt frame and optional actions"""
        self.dit.eval()

        # Move inputs to device and add batch dimension if needed
        prompt = next(iter(test_loader))
        prompt = prompt["video"]
        prompt = prompt[:1, : self.config.n_prompt_frames]  # Use only prompt frames
        if self.config.use_action_conditioning:
            actions = prompt["actions"]
        else:
            actions = None

        # Encode prompt with VAE
        latents = self.encode_frames(prompt)
        batch_size = latents.shape[0]
        n_prompt_frames = latents.shape[1]
        print(f"\nStarting prediction with noise range: {self.noise_range.tolist()}")
        # Generation loop
        print(f"n_prompt_frames: {n_prompt_frames}.. num_frames: {num_frames}")
        for i in tqdm(
            range(n_prompt_frames, num_frames),
            desc="Generating test frames",
            disable=not self.accelerator.is_local_main_process,
        ):
            print(f"\nGenerating frame {i}")

            # Start with pure noise for the new frame
            x_noisy = latents.clone()
            # Add noise to last frame using same noise schedule as training
            ctx_noise = torch.randn_like(x_noisy)
            ctx_noise = torch.clamp(
                ctx_noise, -self.config.noise_abs_max, self.config.noise_abs_max
            )

            # Use stabilization level for context frames like in training
            t_ctx = torch.full(
                (batch_size, latents.shape[1]),
                self.stabilization_level - 1,
                dtype=torch.long,
                device=self.accelerator.device,
            )
            x_noisy = (
                self.alphas_cumprod[t_ctx].sqrt() * x_noisy
                + (1 - self.alphas_cumprod[t_ctx]).sqrt() * ctx_noise
            )

            new_frame = torch.randn(
                (batch_size, 1, *latents.shape[2:]), device=self.accelerator.device
            )
            new_frame = torch.clamp(
                new_frame, -self.config.noise_abs_max, self.config.noise_abs_max
            )

            # Append the new noisy frame to existing context
            x_noisy = torch.cat([x_noisy, new_frame], dim=1)
            print(
                f"Stabilized context frames range: [{x_noisy[:, :-1].min():.4f}, {x_noisy[:, :-1].max():.4f}]"
            )

            # Progressive denoising of the last frame
            for step_idx in reversed(range(self.config.ddim_noise_steps + 1)):
                actual_noise_level = self.noise_range[step_idx]
                next_step_idx = max(0, step_idx - 1)
                next_noise_level = self.noise_range[next_step_idx]

                valid_noise_level = max(0, actual_noise_level)
                valid_next_level = max(0, next_noise_level)

                print(f"\nDenoising step {step_idx}:")
                print(f"Current noise level: {valid_noise_level}")
                print(f"Next noise level: {valid_next_level}")

                # Set noise levels for all frames
                t = torch.full(
                    (batch_size, x_noisy.shape[1]),
                    self.stabilization_level - 1,
                    dtype=torch.long,
                    device=self.accelerator.device,
                )
                t[:, -1] = valid_noise_level

                t_next = t.clone()
                t_next[:, -1] = valid_next_level

                # Get model prediction
                with torch.autocast(
                    "cuda",
                    enabled=True,
                    dtype=torch.bfloat16
                    if self.accelerator.mixed_precision == "bf16"
                    else torch.float16,
                ):
                    print(f"x_noisy: {x_noisy.shape}")
                    print(f"t: {t.shape}")
                    v = self.dit(x_noisy, t, None)
                print(
                    f"Model prediction v range: [{v[:, -1:].min():.4f}, {v[:, -1:].max():.4f}]"
                )

                # Update prediction for last frame only
                t_last = t[:, -1:]
                t_next_last = t_next[:, -1:]
                alpha_t = self.alphas_cumprod[t_last]
                alpha_next = self.alphas_cumprod[t_next_last]

                print(f"alpha_t: {alpha_t.item():.4f}")
                print(f"alpha_next: {alpha_next.item():.4f}")

                x_start = alpha_t.sqrt() * x_noisy[:, -1:] - (1 - alpha_t).sqrt() * v[:, -1:]
                x_noise = ((1 / alpha_t).sqrt() * x_noisy[:, -1:] - x_start) / (1 / alpha_t - 1).sqrt()

                print(f"x_start range: [{x_start.min():.4f}, {x_start.max():.4f}]")

                if step_idx == 0:  # Final step
                    x_pred = x_start
                else:
                    # Use x_noise for next step prediction
                    x_pred = alpha_next.sqrt() * x_start + (1 - alpha_next).sqrt() * x_noise

                print(f"x_pred range: [{x_pred.min():.4f}, {x_pred.max():.4f}]")

                # Update only the last frame
                x_noisy[:, -1:] = x_pred

                # Visualize intermediate steps
                visualize_step(
                    self,
                    x_curr=x_noisy[:1],
                    x_noisy=x_noisy[:1],
                    noise=torch.cat([ctx_noise, new_frame], axis=1)[:1],
                    v=v[:1],
                    pred=torch.cat([x_noisy[:, :-1], x_start], dim=1)[:1],
                    step=global_step,
                    scaling_factor=0.07843137255,
                    name=f"{self.config.model_name}_gen_gs_{global_step}_frame_{i}_step_{step_idx}.png",
                )

            print(
                f"\nFinal frame {i} range: [{x_noisy[:, -1:].min():.4f}, {x_noisy[:, -1:].max():.4f}]"
            )
            # Update the generated frame in latents
            latents = torch.cat([latents, x_noisy[:, -1:]], dim=1)

        # Decode latents to pixels
        scaling_factor = 0.07843137255
        latents = rearrange(latents, "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            pixels = (self.vae.decode(latents / scaling_factor) + 1) / 2
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
        print(f"generation saved to {video_path}.")

        self.dit.train()

    @torch.inference_mode()
    def predict_noise(self, test_loader, epoch, global_step):
        """Generate a video by adding noise to the last frame and then denoising it"""
        self.dit.eval()

        # Get input frames
        prompt = next(iter(test_loader))
        prompt = prompt["video"]
        prompt = prompt[:1]  # Take first batch only

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
        x_noisy[:, :-1] = (
            self.alphas_cumprod[t_ctx].sqrt() * x_noisy[:, :-1]
            + (1 - self.alphas_cumprod[t_ctx]).sqrt() * ctx_noise
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

        # Progressive denoising of the last frame only
        for step_idx in reversed(range(self.config.ddim_noise_steps + 1)):
            actual_noise_level = self.noise_range[step_idx]

            next_step_idx = max(0, step_idx - 1)
            next_noise_level = self.noise_range[next_step_idx]

            # Ensure we use valid indices for alphas_cumprod
            valid_noise_level = max(0, actual_noise_level)  # Can't use negative indices
            valid_next_level = max(0, next_noise_level)

            t = torch.full(
                (batch_size, latents.shape[1]),
                self.stabilization_level - 1,
                dtype=torch.long,
                device=self.accelerator.device,
            )
            t[:, -1] = valid_noise_level

            t_next = t.clone()
            t_next[:, -1] = valid_next_level

            # Print debug info at start
            if step_idx == self.config.ddim_noise_steps:
                print(f"\nInitial noise level: {actual_noise_level}")
                print(
                    f"Initial alpha_t: {self.alphas_cumprod[actual_noise_level].item()}"
                )
                print(f"Noise range steps: {self.noise_range.tolist()}")

            # Get model prediction
            with torch.autocast(
                "cuda",
                enabled=True,
                dtype=torch.bfloat16
                if self.accelerator.mixed_precision == "bf16"
                else torch.float16,
            ):
                v = self.dit(x_noisy, t, None)

            # Update prediction for last frame only
            t_last = t[:, -1:]
            t_next_last = t_next[:, -1:]
            alpha_t = self.alphas_cumprod[t_last]
            alpha_next = self.alphas_cumprod[t_next_last]

            # Use the variance-preserving formulation like in training_step
            x_start = alpha_t.sqrt() * x_noisy[:, -1:] - (1 - alpha_t).sqrt() * v[:, -1:]
            x_noise = ((1 / alpha_t).sqrt() * x_noisy[:, -1:] - x_start) / (1 / alpha_t - 1).sqrt()


            print(f"\nStep {step_idx}:")
            print(
                f"x_noisy range: [{x_noisy[:,-1:].min():.4f}, {x_noisy[:,-1:].max():.4f}]"
            )
            print(f"v range: [{v[:,-1:].min():.4f}, {v[:,-1:].max():.4f}]")
            print(f"alpha_t: {alpha_t.tolist()}")
            print(f"alpha_next: {alpha_next.tolist()}")

            print(f"x_start range: [{x_start.min():.4f}, {x_start.max():.4f}]")

            if step_idx == 0:  # Final step
                x_pred = x_start
            else:
                # Use x_noise for next step prediction
                x_pred = alpha_next.sqrt() * x_start + (1 - alpha_next).sqrt() * x_noise

            print(f"x_pred range: [{x_pred.min():.4f}, {x_pred.max():.4f}]")
            # Visualize intermediate steps
            visualize_step(
                self,
                x_curr=latents[:1],
                x_noisy=x_noisy[:1],
                noise=torch.cat([ctx_noise, noise], dim=1)[:1],  # Make sure noise is properly shaped
                v=v[:1],
                pred=torch.cat([x_noisy[:, :-1], x_pred], dim=1)[
                    :1
                ],  # Keep context frames unchanged
                step=global_step,
                scaling_factor=0.07843137255,
                name=f"{self.config.model_name}_noise_gs_{global_step}_pred_step_{step_idx}.png",
            )

            # Only update the last frame
            x_noisy[:, -1:] = x_pred

        self.dit.train()

    def training_step(self, frames, actions, global_step, visualize: bool = False):
        """Single training step with context-aware noise scheduling"""

        if not hasattr(self, "_first_step_done"):
            rank = self.accelerator.process_index
            world_size = self.accelerator.num_processes
            print(f"[GPU {rank}/{world_size}] Frames shape: {frames.shape}")
            print(
                f"[GPU {rank}/{world_size}] Frame values - Min: {frames.min():.3f}, Max: {frames.max():.3f}, Mean: {frames.mean():.3f}"
            )
            if actions is not None:
                print(f"[GPU {rank}/{world_size}] Actions shape: {actions.shape}")
                print(
                    f"[GPU {rank}/{world_size}] Actions values - Min: {actions.min():.3f}, Max: {actions.max():.3f}, Mean: {actions.mean():.3f}"
                )
            self._first_step_done = True

        self.optimizer.zero_grad()

        batch_size = frames.shape[0]
        total_frames = frames.shape[1]

        # Encode frames to latent space
        latents = self.encode_frames(frames)

        # Initialize loss accumulator
        total_loss = 0.0

        # Process frames sequentially after context frames
        for i in range(self.config.n_prompt_frames, total_frames):  # 1 .. 4
            x_input = latents[:, : i + 1]
            if actions is not None:
                actions_input = actions[:, : i + 1]
            else:
                actions_input = None

            # print(f"x_input: {x_input.shape}")

            # Calculate start frame for sliding window
            start_frame = max(0, i + 1 - self.max_frames)  # Always 0 in our data

            use_max_noise = (  # I keep this for debugging. It's not used in training
                True  # torch.rand(1).item() < 1.2  # 20% chance to use max noise
            )

            if use_max_noise:
                # Use maximum noise level for target frames

                t = torch.full(
                    (batch_size, total_frames),
                    self.noise_range[-1],  # Maximum noise level
                    dtype=torch.long,
                    device=self.accelerator.device,
                )
            else:
                # Random noise levels as before
                t = torch.randint(
                    0,
                    self.noise_range[-1],
                    (batch_size, total_frames),
                    device=self.accelerator.device,
                )

            # Set context frames to stabilization level
            t[:, :-1] = self.stabilization_level - 1

            # print(f"t_next: {t_next.shape}. {t_next}")

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
            x_noisy[:, :-1] = (
                self.alphas_cumprod[t[:, :-1]].sqrt() * x_noisy[:, :-1]
                + (1 - self.alphas_cumprod[t[:, :-1]]).sqrt() * ctx_noise
            )

            # Add noise to current frame
            noise = torch.randn_like(x_curr[:, -1:])
            noise = torch.clamp(
                noise, -self.config.noise_abs_max, self.config.noise_abs_max
            )
            x_noisy[:, -1:] = (
                self.alphas_cumprod[t[:, -1:]].sqrt() * x_curr[:, -1:] +
                (1 - self.alphas_cumprod[t[:, -1:]]).sqrt() * noise
            )

            # Model prediction
            v = self.dit(x_noisy, t, actions_curr)

            # The model predicts v (noise), and we can directly compare it with the noise we added
            loss = nn.functional.mse_loss(v[:, -1:], noise)

            if visualize:
                with torch.no_grad():
                    t_last = t[:, -1:]
                    alpha_t = self.alphas_cumprod[t_last]
                
                    # Calculate x_start and x_noise separately for visualization
                    print(f"Train Reconstruction debug:")
                    print(f"alpha_t: {alpha_t.tolist()}")
                    print(f"alpha_t.sqrt(): {alpha_t.sqrt().tolist()}")

                    print(f"1 - alpha_t: {(1 - alpha_t).tolist()}")
                    print(f"(1 - alpha_t).sqrt(): {(1 - alpha_t).sqrt().tolist()}")

                    print(
                        f"x_noisy range: [{x_noisy[:,-1:].min():.4f}, {x_noisy[:,-1:].max():.4f}]"
                    )
                    
                    x_start = alpha_t.sqrt() * x_noisy[:, -1:] - (1 - alpha_t).sqrt() * v[:, -1:]
                    
                    print(f"v range: [{v[:,-1:].min():.4f}, {v[:,-1:].max():.4f}]")
                    print(f"x_start range: [{x_start.min():.4f}, {x_start.max():.4f}]")
                    print(
                        f"x_curr range: [{x_curr[:, -1:].min():.4f}, {x_curr[:, -1:].max():.4f}]"
                    )

                   
                    #x_noise = ((1 / alpha_t).sqrt() * x_noisy[:, -1:] - x_start) / (1 / alpha_t - 1).sqrt()

                    # Prepare visualization tensor
                    x_recon = torch.zeros_like(x_curr)
                    x_recon[:, :-1] = x_noisy[:, :-1]  # Keep context frames
                    x_recon[:, -1:] = x_start  # Use denoised prediction

                    visualize_step(
                        self,
                        x_curr=x_curr[:1],
                        x_noisy=x_noisy[:1],
                        noise=torch.cat([ctx_noise, noise], axis=1)[:1],
                        v=v[:1],
                        pred=x_recon[:1],
                        step=i,
                        scaling_factor=0.07843137255,
                        name=f"{self.config.model_name}_training_step_{global_step}.png",
                    )

            # Accumulate loss
            total_loss += loss

            # Scale the loss back down for backward pass
            scaled_loss = loss / self.config.gradient_accumulation_steps

            # Backward pass for each frame
            self.accelerator.backward(scaled_loss)

        return total_loss / (total_frames - self.config.n_prompt_frames)

    def save_checkpoint(self, epoch, global_step):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            print(
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
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

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
            val_losses = self.validation(val_loader)
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
                            val_losses = self.validation(val_loader)
                            avg_val_loss = sum(d["loss"] for d in val_losses) / len(
                                val_losses
                            )

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
                            """
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
        print(
            "Using WebDataset. This will stream the dataset from the webdataset directory. Is memory efficient, but may be slow."
        )
        ImageDataset = WebDataset
    elif config.dataset_type == "hfdataset":
        print(
            "Using HFDataset. This will load the dataset into memory. Is faster, but requires A LOT of RAM."
        )
        ImageDataset = HfDataset
    elif config.dataset_type == "dummy":
        print("Using dummy dataset for testing purposes.")
        ImageDataset = DummyDataset
    else:
        raise ValueError(
            f"Invalid dataset type: {config.dataset_type}. Must be 'webdataset' or 'hfdataset'."
        )

    train_loader = DataLoader(
        ImageDataset(split="train", return_actions=config.use_action_conditioning),
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
