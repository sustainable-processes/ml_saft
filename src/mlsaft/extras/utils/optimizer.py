from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from torch.optim import SGD, Adam, NAdam, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)

OPTIMIZER_OPTIONS = Literal["adam", "nadam", "sgd"]
SCHEDULER_OPTIONS = Optional[Literal["step", "cosine"]]


def get_optimizer(
    optimizer_name: OPTIMIZER_OPTIONS, model_params, optimizer_kwargs: Dict[str, Any]
):
    """Get optimizer from name and parameters"""
    if optimizer_name == "adam":
        betas = optimizer_kwargs.get("betas")
        if not betas:
            beta_1 = optimizer_kwargs.get("beta_1", 0.9)
            beta_2 = optimizer_kwargs.get("beta_2", 0.999)
            betas = (beta_1, beta_2)
        optimizer_kwargs["betas"] = betas
        if "beta_1" in optimizer_kwargs:
            del optimizer_kwargs["beta_1"]
        if "beta_2" in optimizer_kwargs:
            del optimizer_kwargs["beta_2"]
        return Adam(params=model_params, **optimizer_kwargs)
    elif optimizer_name == "nadam":
        return NAdam(params=model_params, **optimizer_kwargs)
    elif optimizer_name == "sgd":
        return SGD(params=model_params, **optimizer_kwargs)
    else:
        raise ValueError("Optimizer {} is not supported".format(optimizer_name))


def get_lr_scheduler(
    scheduler_name: SCHEDULER_OPTIONS,
    optimizer: Optimizer,
    scheduler_kwargs: Dict[str, Any],
) -> Union[_LRScheduler, LRScheduler, None]:
    """Get learning rate scheduler from name and parameters"""
    if not scheduler_name:
        return
    elif scheduler_name == "step":
        return StepLR(optimizer=optimizer, **scheduler_kwargs)
    elif scheduler_name == "noam":
        for param, value in scheduler_kwargs.items():
            if not isinstance(value, list):
                scheduler_kwargs[param] = [value]
        if "max_lr_ratio" in scheduler_kwargs:
            scheduler_kwargs["max_lr"] = [
                (
                    scheduler_kwargs.pop("max_lr_ratio")[0]
                    * scheduler_kwargs["init_lr"][0]
                )
            ]
        if "final_lr_ratio" in scheduler_kwargs:
            scheduler_kwargs["final_lr"] = [
                (
                    scheduler_kwargs.pop("final_lr_ratio")[0]
                    * scheduler_kwargs["init_lr"][0]
                )
            ]
        return NoamLR(optimizer=optimizer, **scheduler_kwargs)
    else:
        raise ValueError("Scheduler {} is not supported".format(scheduler_name))


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.

    Copied from Chemprop
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: List[Union[float, int]],
        total_epochs: List[int],
        steps_per_epoch: int,
        init_lr: List[float],
        max_lr: List[float],
        final_lr: List[float],
    ):
        """
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
        :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
        """
        if not (
            len(optimizer.param_groups)
            == len(warmup_epochs)
            == len(total_epochs)
            == len(init_lr)
            == len(max_lr)
            == len(final_lr)
        ):
            raise ValueError(
                "Number of param groups must match the number of epochs and learning rates! "
                f"got: len(optimizer.param_groups)= {len(optimizer.param_groups)}, "
                f"len(warmup_epochs)= {len(warmup_epochs)}, "
                f"len(total_epochs)= {len(total_epochs)}, "
                f"len(init_lr)= {len(init_lr)}, "
                f"len(max_lr)= {len(max_lr)}, "
                f"len(final_lr)= {len(final_lr)}"
            )

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps  # type: ignore

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (
            1 / (self.total_steps - self.warmup_steps)
        )

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Gets a list of the current learning rates.

        :return: A list of the current learning rates.
        """
        return list(self.lr)

    def get_last_lr(self) -> List[float]:
        return super().get_last_lr()

    def step(self, current_step: Optional[int] = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
                             If None, :code:`current_step = self.current_step + 1`.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = (
                    self.init_lr[i] + self.current_step * self.linear_increment[i]
                )
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (
                    self.exponential_gamma[i]
                    ** (self.current_step - self.warmup_steps[i])
                )
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]["lr"] = self.lr[i]
        self._last_lr = self.lr
