import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from model.dit import DiT_models
from model.vae import VAE_models
from utils import one_hot_actions, sigmoid_beta_schedule
from einops import rearrange
from tqdm import tqdm
import wandb
import os
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    vae_checkpoint: str = "checkpoints/vit-l-20.pt"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 4
    num_epochs: int = 100
    save_every: int = 5
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"  # "no" for float32, "fp16" for mixed precision
    seed: int = 42
    use_wandb: bool = True
    output_dir: str = "checkpoints"
    ddim_noise_steps: int = 16
    ctx_max_noise_idx: int = 4  # (ddim_noise_steps // 10) * 3
    noise_abs_max: float = 20.0
    n_prompt_frames: int = 4


class DiffusionTrainer:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="wandb" if config.use_wandb else None
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
            weight_decay=config.weight_decay
        )
        
        # Prepare models, optimizer, and dataloaders with accelerator
        self.dit, self.vae, self.optimizer = self.accelerator.prepare(
            self.dit, self.vae, self.optimizer
        )
        
        # Setup diffusion parameters
        self.max_noise_level = 1000
        self.betas = sigmoid_beta_schedule(self.max_noise_level)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod = rearrange(self.alphas_cumprod, "T -> T 1 1 1")
        
        # Move diffusion parameters to accelerator device
        self.betas, self.alphas, self.alphas_cumprod = self.accelerator.prepare(
            self.betas, self.alphas, self.alphas_cumprod
        )

         # Update diffusion parameters with DDIM-style scheduling
        self.noise_range = torch.linspace(
            -1, self.max_noise_level - 1, config.ddim_noise_steps + 1
        ).long()
        
        # Prepare noise parameters with accelerator
        self.noise_range = self.accelerator.prepare(self.noise_range)
        
    def encode_frames(self, frames):
        """Encode frames using frozen VAE"""
        scaling_factor = 0.07843137255
        frames = rearrange(frames, "b t h w c -> (b t) c h w")
        
        with torch.no_grad():
            latents = self.vae.encode(frames * 2 - 1).mean * scaling_factor
            
        H, W = frames.shape[-2:]
        latents = rearrange(
            latents,
            "(b t) (h w) c -> b t c h w",
            t=frames.shape[1],
            h=H // self.vae.patch_size,
            w=W // self.vae.patch_size,
        )
        return latents
    
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
            x_input = latents[:, :i + 1]
            actions_input = actions[:, :i + 1]
            
            # Calculate start frame for sliding window
            start_frame = max(0, i + 1 - self.dit.max_frames)
            
            # Sample noise indices
            noise_idx = torch.randint(1, self.config.ddim_noise_steps + 1, (1,)).item()
            ctx_noise_idx = min(noise_idx, self.config.ctx_max_noise_idx)
            
            # Prepare noise levels for context and current frame
            t_ctx = torch.full(
                (batch_size, i),
                self.noise_range[ctx_noise_idx],
                device=self.accelerator.device
            )
            t = torch.full(
                (batch_size, 1),
                self.noise_range[noise_idx],
                device=self.accelerator.device
            )
            t_next = torch.full(
                (batch_size, 1),
                self.noise_range[noise_idx - 1],
                device=self.accelerator.device
            )
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)
            
            # Apply sliding window
            x_curr = x_input[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]
            actions_curr = actions_input[:, start_frame:]
            
            # Add noise to context frames
            ctx_noise = torch.randn_like(x_curr[:, :-1])
            ctx_noise = torch.clamp(ctx_noise, -self.config.noise_abs_max, self.config.noise_abs_max)
            x_noisy = x_curr.clone()
            x_noisy[:, :-1] = (
                self.alphas_cumprod[t[:, :-1]].sqrt() * x_noisy[:, :-1] +
                (1 - self.alphas_cumprod[t[:, :-1]]).sqrt() * ctx_noise
            )
            
            # Add noise to current frame
            noise = torch.randn_like(x_curr[:, -1:])
            noise = torch.clamp(noise, -self.config.noise_abs_max, self.config.noise_abs_max)
            x_noisy[:, -1:] = (
                self.alphas_cumprod[t[:, -1:]].sqrt() * x_noisy[:, -1:] +
                (1 - self.alphas_cumprod[t[:, -1:]]).sqrt() * noise
            )
            
            # Model prediction
            v = self.dit(x_noisy, t, actions_curr)
            
            # Compute loss (only on the current frame)
            loss = nn.functional.mse_loss(v[:, -1:], noise)
            
            # Accumulate loss
            total_loss += loss
            
            # Backward pass for each frame
            self.accelerator.backward(loss / (total_frames - self.config.n_prompt_frames))

        return total_loss / (total_frames - self.config.n_prompt_frames)
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.config.output_dir, f"dit_epoch_{epoch+1}.pt"
            )
            self.accelerator.save(
                self.accelerator.unwrap_model(self.dit).state_dict(),
                checkpoint_path
            )
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def train(self, train_loader):
        """Training loop"""
        # Prepare dataloader with accelerator
        train_loader = self.accelerator.prepare(train_loader)
        
        self.dit.train()
        total_training_steps = len(train_loader) * self.config.num_epochs
        
        progress_bar = tqdm(
            total=total_training_steps,
            disable=not self.accelerator.is_local_main_process
        )
        
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            
            for step, batch in enumerate(train_loader):
                frames, actions = batch
                actions = one_hot_actions(actions)
                
                loss = self.training_step(frames, actions)
                epoch_losses.append(loss)
                
                # Update weights after gradient accumulation steps
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    progress_bar.update(1)
                
                # Logging
                if self.accelerator.is_main_process:
                    progress_bar.set_description(
                        f"Epoch {epoch+1} Loss: {sum(epoch_losses) / len(epoch_losses):.4f}"
                    )
                    
                    if self.config.use_wandb:
                        wandb.log({
                            "train_loss": loss,
                            "epoch": epoch,
                            "step": step + epoch * len(train_loader)
                        })
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch)
        
        progress_bar.close()


def main():
    config = TrainingConfig()
    
    trainer = DiffusionTrainer(config)
    
    # Setup data loading
    # Note: Implement your custom Dataset class based on your data format
    train_loader = DataLoader(
        YourDataset(),  # Implement this
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Start training
    trainer.train(train_loader)


if __name__ == "__main__":
    main()