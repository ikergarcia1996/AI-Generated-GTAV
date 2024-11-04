import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from model.vae import VAE_models
from tqdm import tqdm
import wandb
import os
from dataclasses import dataclass


@dataclass
class VAETrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    num_epochs: int = 100
    save_every: int = 5
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    seed: int = 42
    use_wandb: bool = True
    output_dir: str = "checkpoints"
    kl_weight: float = 1e-6  # Weight for KL divergence loss


class VAETrainer:
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
            wandb.init(project="vae-training", config=vars(config))
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Initialize VAE model
        self.vae = VAE_models["vit-l-20-shallow-encoder"]()
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.vae.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Prepare model and optimizer with accelerator
        self.vae, self.optimizer = self.accelerator.prepare(
            self.vae, self.optimizer
        )

    def training_step(self, frames):
        """Single training step"""
        # Forward pass
        recon, posterior, _ = self.vae(frames, None)
        
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon, frames)
        
        # KL divergence loss
        kl_loss = 0.5 * torch.sum(
            posterior.mean**2 + posterior.var - 1.0 - torch.log(posterior.var + 1e-8),
            dim=posterior.dims
        ).mean()
        
        # Total loss
        loss = recon_loss + self.config.kl_weight * kl_loss
        
        # Backward pass
        self.accelerator.backward(loss)
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.config.output_dir, f"vae_epoch_{epoch+1}.pt"
            )
            self.accelerator.save(
                self.accelerator.unwrap_model(self.vae).state_dict(),
                checkpoint_path
            )
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def train(self, train_loader):
        """Training loop"""
        train_loader = self.accelerator.prepare(train_loader)
        
        self.vae.train()
        total_training_steps = len(train_loader) * self.config.num_epochs
        
        progress_bar = tqdm(
            total=total_training_steps,
            disable=not self.accelerator.is_local_main_process
        )
        
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            
            for step, batch in enumerate(train_loader):
                frames = batch[0]  # Assuming batch returns (frames, _)
                
                # Training step
                loss_dict = self.training_step(frames)
                epoch_losses.append(loss_dict['loss'])
                
                # Update weights
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
                            'train_loss': loss_dict['loss'],
                            'recon_loss': loss_dict['recon_loss'],
                            'kl_loss': loss_dict['kl_loss'],
                            'epoch': epoch,
                            'step': step + epoch * len(train_loader)
                        })
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch)
        
        progress_bar.close()


def main():
    config = VAETrainingConfig()
    
    trainer = VAETrainer(config)
    
    # Setup data loading
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