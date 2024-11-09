import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from model.dit import DiT_models
from model.vae import VAE_models
from utils import sigmoid_beta_schedule
from einops import rearrange
from tqdm import tqdm
import wandb
import os
from dataclasses import dataclass
import yaml
from dataset import ImageDataset, split_len
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup


@dataclass
class TrainingConfig:
    vae_checkpoint: str = "checkpoints/vit-l-20.pt"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    num_epochs: int = 100
    save_every: int = 5
    gradient_accumulation_steps: int = 1
    seed: int = 42
    use_wandb: bool = True
    output_dir: str = "checkpoints"
    ddim_noise_steps: int = 16
    ctx_max_noise_idx: int = 4  # (ddim_noise_steps // 10) * 3
    noise_abs_max: float = 20.0
    n_prompt_frames: int = 4
    min_learning_rate: float = 1e-6
    validation_batch_size: int = 8
    max_steps: int = -1  # -1 means no maximum steps limit
    validation_steps: int = 1000
    logging_steps: int = 10
    use_action_conditioning: bool = True
    warnup_ratio: float = 0.1

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
        )

        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(project="diffusion-transformer", config=vars(config))

        # Set seed for reproducibility
        set_seed(config.seed)

        # Initialize models
        self.dit = DiT_models["DiT-S/2"]()
        self.vae = VAE_models["vit-l-20-shallow-encoder"]()

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
        )

        # Calculate total steps for scheduler
        total_dataset_size = split_len("train") * 5
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
            num_cycles=0.5,  # Standard cosine decay
            min_lr=self.config.min_learning_rate,
        )

        # Update prepare statement to include scheduler
        self.dit, self.vae, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.dit, self.vae, self.optimizer, self.scheduler
        )

        # Setup diffusion parameters
        self.max_noise_level = 1000
        self.betas = sigmoid_beta_schedule(self.max_noise_level)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod = rearrange(self.alphas_cumprod, "T -> T 1 1 1")

        self.betas = self.betas.to(self.accelerator.device)
        self.alphas = self.alphas.to(self.accelerator.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.accelerator.device)

        # Update diffusion parameters with DDIM-style scheduling
        self.noise_range = (
            torch.linspace(-1, self.max_noise_level - 1, config.ddim_noise_steps + 1)
            .long()
            .to(self.accelerator.device)
        )

        # Move diffusion parameters to accelerator device
        # self.betas, self.alphas, self.alphas_cumprod = self.accelerator.prepare(
        #    self.betas, self.alphas, self.alphas_cumprod
        # )

        # Update diffusion parameters with DDIM-style scheduling
        self.noise_range = torch.linspace(
            -1, self.max_noise_level - 1, config.ddim_noise_steps + 1
        ).long()

        # Prepare noise parameters with accelerator
        self.noise_range = self.accelerator.prepare(self.noise_range)

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

        with tqdm(total=total_steps, desc="Validation") as pbar:
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
                    start_frame = max(0, i + 1 - self.dit.max_frames)

                    noise_idx = torch.randint(
                        1, self.config.ddim_noise_steps + 1, (1,)
                    ).item()
                    ctx_noise_idx = min(noise_idx, self.config.ctx_max_noise_idx)

                    # Prepare noise levels for context and current frame
                    t_ctx = torch.full(
                        (batch_size, i),
                        self.noise_range[ctx_noise_idx],
                        device=self.accelerator.device,
                    )
                    t = torch.full(
                        (batch_size, 1),
                        self.noise_range[noise_idx],
                        device=self.accelerator.device,
                    )
                    t_next = torch.full(
                        (batch_size, 1),
                        self.noise_range[noise_idx - 1],
                        device=self.accelerator.device,
                    )
                    t_next = torch.where(t_next < 0, t, t_next)
                    t = torch.cat([t_ctx, t], dim=1)
                    t_next = torch.cat([t_ctx, t_next], dim=1)

                    # Apply sliding window
                    x_curr = x_input[:, start_frame:]
                    t = t[:, start_frame:]
                    t_next = t_next[:, start_frame:]
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
                        self.alphas_cumprod[t[:, -1:]].sqrt() * x_noisy[:, -1:]
                        + (1 - self.alphas_cumprod[t[:, -1:]]).sqrt() * noise
                    )

                    # Model prediction
                    v = self.dit(x_noisy, t, actions_curr)

                    # Compute loss (only on the current frame)
                    loss = nn.functional.mse_loss(v[:, -1:], noise)
                    total_loss += loss

                avg_loss = total_loss / (total_frames - self.config.n_prompt_frames)
                val_losses.append({"loss": avg_loss.item()})
                pbar.update(1)

        self.dit.train()
        return val_losses

    def training_step(self, frames, actions):
        """Single training step with context-aware noise scheduling"""
        self.optimizer.zero_grad()

        batch_size = frames.shape[0]
        total_frames = frames.shape[1]

        # Encode frames to latent space
        latents = self.encode_frames(frames)

        # Initialize loss accumulator
        total_loss = 0.0

        # Process frames sequentially after context frames
        for i in range(self.config.n_prompt_frames, total_frames):
            # Get input frames up to current frame
            x_input = latents[:, : i + 1]
            if actions is not None:
                actions_input = actions[:, : i + 1]
            else:
                actions_input = None

            # Calculate start frame for sliding window
            start_frame = max(0, i + 1 - self.dit.max_frames)

            # Sample noise indices
            noise_idx = torch.randint(1, self.config.ddim_noise_steps + 1, (1,)).item()
            ctx_noise_idx = min(noise_idx, self.config.ctx_max_noise_idx)

            # Prepare noise levels for context and current frame
            t_ctx = torch.full(
                (batch_size, i),
                self.noise_range[ctx_noise_idx],
                device=self.accelerator.device,
            )
            t = torch.full(
                (batch_size, 1),
                self.noise_range[noise_idx],
                device=self.accelerator.device,
            )
            t_next = torch.full(
                (batch_size, 1),
                self.noise_range[noise_idx - 1],
                device=self.accelerator.device,
            )
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            # Apply sliding window
            x_curr = x_input[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]
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
                self.alphas_cumprod[t[:, -1:]].sqrt() * x_noisy[:, -1:]
                + (1 - self.alphas_cumprod[t[:, -1:]]).sqrt() * noise
            )

            # Model prediction
            v = self.dit(x_noisy, t, actions_curr)
            
            # Ensure tensors have matching shapes before MSE loss
            if v[:, -1:].shape != noise.shape:
                # Option 1: Resize noise to match v
                noise = noise[..., :v.shape[-1]]
                # OR Option 2: Resize v to match noise
                # v = v[..., :noise.shape[-1]]
            
            loss = nn.functional.mse_loss(v[:, -1:], noise)

            # Accumulate loss
            total_loss += loss

            # Backward pass for each frame
            self.accelerator.backward(
                loss / (total_frames - self.config.n_prompt_frames)
            )

        return total_loss / (total_frames - self.config.n_prompt_frames)

    def save_checkpoint(self, epoch, global_step):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            print(
                f"Saving checkpoint at epoch {epoch+1} and step {global_step} to {self.config.output_dir}"
            )
            os.makedirs(self.config.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.config.output_dir, f"dit_epoch_{epoch+1}_{global_step}.pt"
            )
            # Fix: Save DiT model state dict instead of VAE
            self.accelerator.save(
                self.accelerator.unwrap_model(self.dit).state_dict(), checkpoint_path
            )
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def train(self, train_loader, val_loader):
        """Training loop"""
        # Prepare dataloader with accelerator
        train_loader, val_loader = self.accelerator.prepare(train_loader, val_loader)

        self.dit.train()
        total_dataset_size = split_len("train") * 5
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

            for epoch in range(self.config.num_epochs):
                epoch_losses = []

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

                    loss = self.training_step(frames, actions)
                    epoch_losses.append(loss)

                    # Update weights after gradient accumulation steps
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
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
                                    "train_loss": loss,
                                    "learning_rate": current_lr,
                                    "epoch": epoch,
                                    "step": global_step,
                                }
                            )
                    if (
                        global_step > 0
                        and global_step % self.config.validation_steps == 0
                    ):
                        val_losses = self.validation(val_loader)
                        avg_val_loss = sum(d["loss"] for d in val_losses) / len(
                            val_losses
                        )

                        if self.accelerator.is_main_process and self.config.use_wandb:
                            wandb.log(
                                {
                                    "val_loss": avg_val_loss,
                                    "epoch": epoch,
                                    "step": global_step,
                                }
                            )

                    # Save checkpoint
                    if global_step > 0 and global_step % self.config.save_every == 0:
                        self.save_checkpoint(epoch, global_step)

            progress_bar.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config YAML file")
    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config)
    trainer = DiffusionTrainer(config)

    # Setup data loading
    train_loader = DataLoader(
        ImageDataset(split="train", return_actions=config.use_action_conditioning),
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        ImageDataset(split="validation", return_actions=config.use_action_conditioning),
        batch_size=config.validation_batch_size,
        num_workers=0,
        pin_memory=True,
    )

    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
