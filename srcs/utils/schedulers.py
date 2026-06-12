"""
Learning rate schedulers for model training.
"""

import torch.optim


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Noam learning rate scheduler from "Attention is All You Need"

    Args:
        optimizer: torch optimizer
        model_size: dimension of model (typically d_model)
        warmup_steps: number of warmup steps
        factor: scaling factor for learning rate
        last_epoch: epoch to start from
    """

    def __init__(
        self, optimizer, model_size, warmup_steps=4000, factor=1.0, last_epoch=-1
    ):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)
        return [
            self.factor
            * (
                self.model_size ** (-0.5)
                * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
            )
            for _ in self.base_lrs
        ]
