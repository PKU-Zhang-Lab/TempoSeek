"""
Unified training entry for TempoNet and TempoCoder models.
Uses Hydra for configuration management.

Usage:
    python srcs/train.py                                    # Train with default config
    python srcs/train.py model=TempoNet                    # Train TempoNet
    python srcs/train.py model=TempoCoder                  # Train TempoCoder
    python srcs/train.py experiment=temponet_d128l3        # Use specific experiment
    python srcs/train.py model.d_hidden=256               # Override parameters
    python srcs/train.py --multirun model=TempoNet,TempoCoder  # Train both
"""

import hydra
import importlib
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    Main training function

    Args:
        cfg: Hydra configuration object
    """

    print("\n" + "=" * 80)
    print("TempoSeek Training Framework")
    print("=" * 80)
    print("\nTraining Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80 + "\n")

    # Set random seed for reproducibility
    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    # Get model type
    model_type = cfg.model.type
    print(f"[INFO] Loading model: {model_type}")

    # Dynamically import model module
    try:
        # Get absolute path relative to this script
        scripts_dir = Path(__file__).parent
        model_dir_name = cfg.model.dir
        model_dir = scripts_dir / "model" / model_dir_name

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Add model dir to Python path temporarily
        import sys

        # Add srcs directory to path (for common, config imports)
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        model_base_dir = scripts_dir / "model"
        if str(model_base_dir) not in sys.path:
            sys.path.insert(0, str(model_base_dir))

        # Import model module from the specific model folder
        model_module = importlib.import_module(model_dir_name)

        # Get the model and dataloader classes
        ModelClass = getattr(model_module, model_type)  # e.g., TempoNet, TempoCoder
        DataLoaderClass = getattr(
            model_module, f"{model_type}DataLoader"
        )  # e.g., TempoNetDataLoader

        print(
            f"[INFO] Successfully loaded {ModelClass.__name__} and {DataLoaderClass.__name__}"
        )

    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Failed to load model '{model_type}'. Make sure {model_type}/ directory exists "
            f"with __init__.py exporting {model_type} and {model_type}DataLoader\n"
            f"Error: {e}"
        )

    # Create data loaders
    print("\n[INFO] Creating data loaders...")
    print(f"  Train folds: {cfg.data.train.folds}")
    print(f"  Valid folds: {cfg.data.valid.folds}")

    data_loader_config = OmegaConf.to_container(cfg.data, resolve=True)
    same_loader_config = data_loader_config.copy()
    same_loader_config.pop("train", None)
    same_loader_config.pop("valid", None)
    train_loader_config = {**same_loader_config, **data_loader_config.get("train", {})}
    valid_loader_config = {**same_loader_config, **data_loader_config.get("valid", {})}

    # train_loader_config = OmegaConf.to_container(cfg.data.train, resolve=True)
    # valid_loader_config = OmegaConf.to_container(cfg.data.valid, resolve=True)

    train_loader = DataLoaderClass(**train_loader_config)
    valid_loader = DataLoaderClass(**valid_loader_config)

    print(f"[INFO] Train loader: {len(train_loader)} batches")
    print(f"[INFO] Valid loader: {len(valid_loader)} batches")

    # Create model
    print(f"\n[INFO] Creating {model_type} model...")

    # Remove 'type' from model config before passing to model constructor
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config.pop("type", None)
    model_config.pop("dir", None)

    model = ModelClass(**model_config)
    print("[INFO] Model created successfully")

    # Get Hydra runtime info for output directory
    try:
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg.runtime.output_dir
    except (ImportError, AttributeError):
        # Fallback if Hydra config is not available
        output_dir = cfg.log_dir

    print(f"\n[INFO] Output directory: {output_dir}")

    # Setup logger
    print("[INFO] Setting up TensorBoard logger...")
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=cfg.experiment_name or model_type,
        version="",
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/ckpt",
        filename=f"{model_type}-{{epoch:02d}}",
        save_top_k=3,
        monitor="loss/train_loss",
        mode="min",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_callback, lr_monitor]

    # Create Trainer
    print("[INFO] Creating PyTorch Lightning Trainer...")
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        logger=logger,
        callbacks=callbacks,
    )

    # Start training
    print(f"\n[INFO] Starting training for {cfg.trainer.max_epochs} epochs...")
    
    # Check for checkpoint resuming
    ckpt_path = cfg.trainer.get("ckpt_path", None)
    if ckpt_path:
        print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
    
    print("=" * 80 + "\n")

    trainer.fit(model, train_loader, valid_loader, ckpt_path=ckpt_path)

    print("\n" + "=" * 80)
    print("[INFO] Training completed successfully!")
    print(f"[INFO] Checkpoints saved to: {output_dir}/ckpt")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    train()
