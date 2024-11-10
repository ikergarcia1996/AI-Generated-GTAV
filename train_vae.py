import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from einops import rearrange
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

import wandb
from hf_dataset import ImageDataset as HfDataset
from model.vae import VAE_models
from web_dataset import ImageDataset as WebDataset
from web_dataset import split_len


@dataclass
class VAETrainingConfig:
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    batch_size: int = 8
    validation_batch_size: int = 8
    num_epochs: int = 100
    save_every: int = 5
    gradient_accumulation_steps: int = 1
    seed: int = 42
    use_wandb: bool = True
    output_dir: str = "checkpoints"
    kl_weight: float = 1e-6  # Weight for KL divergence loss
    max_steps: int = -1  # -1 means no maximum steps limit
    validation_steps: int = 1000
    logging_steps: int = 10
    warnup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    use_ema: bool = True  # Add EMA (exponential moving average)
    ema_decay: float = 0.995

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "VAETrainingConfig":
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        data = cls(**config_dict)
        data.learning_rate = float(data.learning_rate)
        data.min_learning_rate = float(data.min_learning_rate)
        data.weight_decay = float(data.weight_decay)
        data.kl_weight = float(data.kl_weight)
        data.warnup_ratio = float(data.warnup_ratio)

        return data


class VAETrainer:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__)

        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb" if config.use_wandb else None,
        )

        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(project="vae-training", config=vars(config))

        # Set seed for reproducibility
        set_seed(config.seed)

        # Initialize VAE model
        self.vae = VAE_models["vit-l-20-shallow-encoder"]()

        # Setup optimizer
        self.optimizer = AdamW(
            self.dit.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-7,
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
            num_cycles=0.25,  # Standard cosine decay
            min_lr=self.config.min_learning_rate,
        )

        if config.use_ema:
            from torch_ema import ExponentialMovingAverage

            self.ema = ExponentialMovingAverage(
                self.dit.parameters(), decay=config.ema_decay
            )
            self.ema = self.accelerator.prepare(self.ema)

        # Prepare model, optimizer and scheduler with accelerator
        self.vae, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.vae, self.optimizer, self.scheduler
        )

        self.vae = torch.compile(self.vae)

    def training_step(self, frames):
        """Single training step"""
        # Forward pass
        # Normalize input to [-1, 1] range, matching generation pipeline
        frames_normalized = frames * 2 - 1

        # Forward pass
        recon, posterior, _ = self.vae(frames_normalized, None)

        # Denormalize reconstruction back to [0, 1] range for loss calculation
        recon = (recon + 1) / 2

        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon, frames)

        # KL divergence loss
        kl_loss = (
            0.5
            * torch.sum(
                torch.exp(posterior.logvar)
                + posterior.mean**2
                - 1.0
                - posterior.logvar,
                dim=posterior.dims,
            ).mean()
        )

        # Total loss
        loss = recon_loss + self.config.kl_weight * kl_loss

        # Scale loss by gradient accumulation steps
        loss = loss / self.config.gradient_accumulation_steps
        recon_loss = recon_loss / self.config.gradient_accumulation_steps
        kl_loss = kl_loss / self.config.gradient_accumulation_steps

        # Backward pass
        self.accelerator.backward(loss)

        if self.config.use_ema:
            self.ema.update()

        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    @torch.inference_mode()
    def validation(self, val_loader):
        """Run validation loop"""
        self.vae.eval()
        val_losses = []
        total_dataset_size = split_len("validation") * 5
        total_steps = total_dataset_size // (
            self.config.validation_batch_size * self.accelerator.num_processes
        )
        with torch.autocast(
            "cuda",
            enabled=True,
            dtype=torch.bfloat16
            if self.accelerator.mixed_precision == "bf16"
            else torch.float16,
        ):
            with tqdm(
                total=total_steps,
                desc="Validation",
                disable=not self.accelerator.is_local_main_process,
            ) as pbar:
                for batch in val_loader:
                    frames = rearrange(batch["video"], "b t c h w -> (b t) c h w")
                    frames_normalized = frames * 2 - 1

                    recon, posterior, _ = self.vae(frames_normalized, None)
                    recon = (recon + 1) / 2

                    recon_loss = nn.functional.mse_loss(recon, frames)
                    kl_loss = (
                        0.5
                        * torch.sum(
                            torch.exp(posterior.logvar)
                            + posterior.mean**2
                            - 1.0
                            - posterior.logvar,
                            dim=posterior.dims,
                        ).mean()
                    )

                    loss = recon_loss + self.config.kl_weight * kl_loss
                    val_losses.append(
                        {
                            "loss": loss.item(),
                            "recon_loss": recon_loss.item(),
                            "kl_loss": kl_loss.item(),
                        }
                    )
                    pbar.update(1)

        self.vae.train()
        return val_losses

    def save_checkpoint(self, epoch, global_step):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            print(
                f"Saving checkpoint at epoch {epoch+1} and step {global_step} to {self.config.output_dir}"
            )
            os.makedirs(self.config.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.config.output_dir, f"vae_epoch_{epoch+1}_{global_step}.pt"
            )
            self.accelerator.save(
                self.accelerator.unwrap_model(self.vae).state_dict(), checkpoint_path
            )
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def train(self, train_loader, val_loader):
        """Training loop"""
        train_loader, val_loader = self.accelerator.prepare(train_loader, val_loader)

        self.vae.train()
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
                accumulated_loss_dict = {"loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}

                for step, batch in enumerate(train_loader):
                    if (
                        self.config.max_steps > 0
                        and global_step >= self.config.max_steps
                    ):
                        return

                    frames = rearrange(
                        batch["video"],
                        "b t c h w -> (b t) c h w",
                        b=self.config.batch_size,
                        t=5,
                        c=3,
                        h=360,
                        w=640,
                    )

                    # Training step
                    loss_dict = self.training_step(frames)

                    # Accumulate losses
                    for k, v in loss_dict.items():
                        accumulated_loss_dict[k] += v.item()

                    # Update weights and log
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        # Average the accumulated losses
                        avg_loss_dict = {
                            k: v / self.config.gradient_accumulation_steps
                            for k, v in accumulated_loss_dict.items()
                        }
                        epoch_losses.append(avg_loss_dict["loss"])

                        self.accelerator.clip_grad_norm_(
                            self.dit.parameters(), self.config.max_grad_norm
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        progress_bar.update(1)
                        global_step += 1

                        # Logging
                        if (
                            self.config.use_wandb
                            and global_step % self.config.logging_steps == 0
                        ):
                            current_lr = self.scheduler.get_last_lr()[0]
                            wandb.log(
                                {
                                    "train_loss": avg_loss_dict["loss"],
                                    "recon_loss": avg_loss_dict["recon_loss"],
                                    "kl_loss": avg_loss_dict["kl_loss"],
                                    "learning_rate": current_lr,
                                    "epoch": epoch,
                                    "step": global_step,
                                }
                            )

                        # Reset accumulated losses
                        accumulated_loss_dict = {
                            "loss": 0.0,
                            "recon_loss": 0.0,
                            "kl_loss": 0.0,
                        }

                        progress_bar.set_description(
                            f"Epoch {epoch+1} Loss: {sum(epoch_losses) / len(epoch_losses):.4f}"
                        )

                        # Run validation
                        if (
                            global_step > 0
                            and global_step % self.config.validation_steps == 0
                        ):
                            val_losses = self.validation(val_loader)

                            # Calculate average validation metrics
                            avg_val_metrics = {
                                k: sum(d[k] for d in val_losses) / len(val_losses)
                                for k in val_losses[0].keys()
                            }

                            if (
                                self.accelerator.is_main_process
                                and self.config.use_wandb
                            ):
                                wandb.log(
                                    {
                                        "val_loss": avg_val_metrics["loss"],
                                        "val_recon_loss": avg_val_metrics["recon_loss"],
                                        "val_kl_loss": avg_val_metrics["kl_loss"],
                                        "epoch": epoch,
                                        "step": global_step,
                                    }
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

    config = VAETrainingConfig.from_yaml(args.config)
    trainer = VAETrainer(config)

    # Setup data loading
    if config.dataset_type == "webdataset":
        ImageDataset = WebDataset
    elif config.dataset_type == "hfdataset":
        ImageDataset = HfDataset
    else:
        raise ValueError(
            f"Invalid dataset type: {config.dataset_type}. Must be 'webdataset' or 'hfdataset'."
        )

    train_loader = DataLoader(
        ImageDataset(split="train", return_actions=False),
        batch_size=config.batch_size,
        num_workers=min(os.cpu_count(), 16),
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        ImageDataset(split="validation", return_actions=False),
        batch_size=config.validation_batch_size,
        num_workers=min(os.cpu_count(), 4),
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )

    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
