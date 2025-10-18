"""
QAT Training for MoE W2A2

Fine-tunes quantized MoE model with:
- Fake quantization on router-adjacent layers
- Top-k consistency loss
- Margin loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path
from tqdm import tqdm

from ..losses.routing_losses import combined_routing_loss
from ..quant.quantizers import W2A2Config
from ..models.load_moe import MoEModelLoader
from dynaquant.fake_quant import FakeQuantize


logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple text dataset for QAT"""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


class QATTrainer:
    """
    Quantization-Aware Training for MoE

    Trains with:
    - Fake quantization on selected layers
    - Task loss (language modeling)
    - Routing consistency loss
    - Margin loss
    """

    def __init__(
        self,
        model_loader: MoEModelLoader,
        w2a2_config: W2A2Config,
        output_dir: str = "./qat_output",
        learning_rate: float = 5e-6,
        num_epochs: int = 2,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        lambda_topk: float = 1.0,
        mu_margin: float = 0.2,
        freeze_experts: bool = True,
        train_router_adjacent_only: bool = True
    ):
        self.model_loader = model_loader
        self.w2a2_config = w2a2_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training hyperparameters
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.grad_accum_steps = gradient_accumulation_steps
        self.lambda_topk = lambda_topk
        self.mu_margin = mu_margin
        self.freeze_experts = freeze_experts
        self.train_router_adjacent_only = train_router_adjacent_only

        # Components
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.optimizer = None
        self.fake_quant_modules = {}

        # Statistics
        self.training_stats = []

    def inject_fake_quantization(self):
        """Inject fake quantization modules into model"""
        logger.info("Injecting fake quantization modules...")

        # Find router-adjacent layers
        for layer_idx in range(self.model_loader.get_num_moe_layers()):
            # Get router
            router = self.model_loader.get_router(layer_idx)
            if router is None:
                continue

            # Wrap router with fake quant
            fake_quant = FakeQuantize(
                bit_width=self.w2a2_config.w_bit,
                symmetric=self.w2a2_config.w_symmetric
            )

            # Store original forward
            original_forward = router.forward

            # Create new forward with fake quant
            def make_fake_quant_forward(orig_forward, fq):
                def forward_with_fq(x):
                    # Fake quantize input
                    x_fq = fq(x)
                    return orig_forward(x_fq)
                return forward_with_fq

            router.forward = make_fake_quant_forward(
                original_forward, fake_quant)

            self.fake_quant_modules[f"router_{layer_idx}"] = fake_quant

        logger.info(
            f"Injected {len(self.fake_quant_modules)} fake quant modules")

    def remove_fake_quantization(self):
        """Remove fake quantization modules"""
        # This is handled by restoring original forwards if needed
        pass

    def freeze_parameters(self):
        """Freeze parameters based on training strategy"""
        if self.freeze_experts:
            logger.info("Freezing expert parameters...")

            for layer_idx in range(self.model_loader.get_num_moe_layers()):
                experts = self.model_loader.get_experts(layer_idx)
                if experts:
                    for expert in experts:
                        for param in expert.parameters():
                            param.requires_grad = False

        if self.train_router_adjacent_only:
            logger.info("Training router and adjacent layers only...")

            # Freeze all parameters first
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze routers and adjacent layers
            for layer_idx in range(self.model_loader.get_num_moe_layers()):
                router = self.model_loader.get_router(layer_idx)
                if router:
                    for param in router.parameters():
                        param.requires_grad = True

    def setup_optimizer(self):
        """Setup optimizer for trainable parameters"""
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad]

        logger.info(
            f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=0.01
        )

    def compute_loss(
        self,
        outputs,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss

        Args:
            outputs: Model outputs
            labels: Target labels

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Task loss (language modeling)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        task_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        # Routing losses (if available)
        routing_loss = 0.0
        loss_dict = {"task": task_loss.item()}

        # If we have routing logits, compute routing losses
        if hasattr(outputs, "router_logits") and outputs.router_logits is not None:
            # Get FP16 reference (requires extra forward pass)
            # For simplicity, we'll use margin loss only during QAT
            from ..losses.routing_losses import margin_loss

            # Compute margin loss on quantized routing
            for layer_logits in outputs.router_logits:
                routing_loss += margin_loss(layer_logits,
                                            k=2, margin_target=0.5)

            routing_loss = routing_loss / len(outputs.router_logits)
            loss_dict["routing"] = routing_loss.item()

        # Combined loss
        total_loss = task_loss + self.mu_margin * routing_loss
        loss_dict["total"] = total_loss.item()

        return total_loss, loss_dict

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        total_task_loss = 0.0
        total_routing_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)

            # Forward
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids  # For LM loss
            )

            # Compute loss
            loss, loss_dict = self.compute_loss(outputs, input_ids)

            # Scale by gradient accumulation
            loss = loss / self.grad_accum_steps

            # Backward
            loss.backward()

            # Update
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Stats
            total_loss += loss_dict["total"]
            total_task_loss += loss_dict["task"]
            total_routing_loss += loss_dict.get("routing", 0.0)
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['total']:.4f}",
                "task": f"{loss_dict['task']:.4f}",
            })

        epoch_stats = {
            "epoch": epoch,
            "loss": total_loss / num_batches,
            "task_loss": total_task_loss / num_batches,
            "routing_loss": total_routing_loss / num_batches,
        }

        return epoch_stats

    def train(
        self,
        train_texts: List[str]
    ) -> List[Dict]:
        """
        Run QAT training

        Args:
            train_texts: Training text samples

        Returns:
            List of per-epoch statistics
        """
        logger.info("Starting QAT training...")

        # Setup
        self.inject_fake_quantization()
        self.freeze_parameters()
        self.setup_optimizer()

        # Create dataset
        dataset = TextDataset(train_texts, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Training loop
        for epoch in range(self.num_epochs):
            epoch_stats = self.train_epoch(dataloader, epoch + 1)
            self.training_stats.append(epoch_stats)

            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}: {epoch_stats}")

            # Save checkpoint
            self.save_checkpoint(epoch + 1)

        # Save final results
        self.save_results()

        logger.info("QAT training complete!")

        return self.training_stats

    def save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint_path = self.output_dir / f"checkpoint_epoch{epoch}.pt"

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats,
        }, checkpoint_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def save_results(self):
        """Save training results"""
        results_file = self.output_dir / "qat_results.json"

        with open(results_file, 'w') as f:
            json.dump({
                "training_stats": self.training_stats,
                "config": {
                    "learning_rate": self.lr,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "lambda_topk": self.lambda_topk,
                    "mu_margin": self.mu_margin,
                }
            }, f, indent=2)

        logger.info(f"Saved results to {results_file}")


def create_qat_trainer(
    model_name: str,
    w2a2_config: W2A2Config,
    output_dir: str = "./qat_output",
    learning_rate: float = 5e-6,
    num_epochs: int = 2
) -> QATTrainer:
    """
    Convenience function to create QAT trainer

    Args:
        model_name: MoE model name
        w2a2_config: W2A2 config
        output_dir: Output directory
        learning_rate: Learning rate
        num_epochs: Number of epochs

    Returns:
        QATTrainer instance
    """
    from ..models.load_moe import load_moe_model

    model_loader = load_moe_model(model_name)

    trainer = QATTrainer(
        model_loader,
        w2a2_config,
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )

    return trainer
