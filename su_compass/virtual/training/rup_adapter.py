"""Optional FedProx and utility instrumentation without changing APPFL trainers.

The adapter is deliberately scoped to one ``client_agent.train()`` call.  When
both features are disabled it calls the original agent directly, which keeps
FedCompass baselines byte-for-byte on the old training path.
"""

from __future__ import annotations

import math
import random
import types
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class RUPTrainingObservation:
    client_id: str
    local_steps: int
    num_train_samples: int
    loss_before: Optional[float]
    loss_after: Optional[float]
    loss_delta: Optional[float]
    loss_delta_per_step: Optional[float]
    prox_mu: float
    mean_prox_penalty: float
    mean_base_loss: float
    num_observed_batches: int
    finite: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RUPTrainingAdapter:
    """Inject a proximal objective and collect comparable local-loss signals."""

    def __init__(
        self,
        *,
        utility_enabled: bool = False,
        prox_enabled: bool = False,
        prox_mu: float = 1e-4,
        utility_eval_batches: int = 1,
    ) -> None:
        if prox_mu < 0:
            raise ValueError("prox_mu must be non-negative")
        if utility_eval_batches < 1:
            raise ValueError("utility_eval_batches must be positive")
        self.utility_enabled = bool(utility_enabled)
        self.prox_enabled = bool(prox_enabled and prox_mu > 0)
        self.prox_mu = float(prox_mu if self.prox_enabled else 0.0)
        self.utility_eval_batches = int(utility_eval_batches)

    def train(self, client_agent, client_id: str, local_steps: int) -> RUPTrainingObservation:
        trainer = client_agent.trainer
        num_samples = len(trainer.train_dataset)
        if not self.utility_enabled and not self.prox_enabled:
            client_agent.train()
            return RUPTrainingObservation(
                client_id=client_id, local_steps=local_steps,
                num_train_samples=num_samples, loss_before=None, loss_after=None,
                loss_delta=None, loss_delta_per_step=None, prox_mu=0.0,
                mean_prox_penalty=0.0, mean_base_loss=0.0,
                num_observed_batches=0, finite=True,
            )

        loss_before = self._evaluate_fixed_batches(trainer) if self.utility_enabled else None
        reference = {
            name: parameter.detach().cpu().clone()
            for name, parameter in trainer.model.named_parameters()
        }
        original_train_batch = trainer._train_batch
        totals = {"prox": 0.0, "base": 0.0, "batches": 0}

        if self.prox_enabled:
            def prox_train_batch(bound_trainer, optimizer, data, target):
                device = bound_trainer.train_configs.device
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = bound_trainer.model(data)
                base_loss = bound_trainer.loss_fn(output, target)
                prox_sq = torch.zeros((), device=base_loss.device)
                for name, parameter in bound_trainer.model.named_parameters():
                    ref = reference[name].to(parameter.device, non_blocking=True)
                    prox_sq = prox_sq + torch.sum((parameter - ref) ** 2)
                prox_penalty = 0.5 * self.prox_mu * prox_sq
                total_loss = base_loss + prox_penalty
                total_loss.backward()
                if (
                    bound_trainer.train_configs.get("clip_grad", False)
                    or bound_trainer.train_configs.get("use_dp", False)
                ):
                    torch.nn.utils.clip_grad_norm_(
                        bound_trainer.model.parameters(),
                        bound_trainer.train_configs.clip_value,
                        norm_type=bound_trainer.train_configs.clip_norm,
                    )
                optimizer.step()
                totals["prox"] += float(prox_penalty.detach().cpu())
                totals["base"] += float(base_loss.detach().cpu())
                totals["batches"] += 1
                return (
                    float(base_loss.detach().cpu()),
                    output.detach().cpu().numpy(),
                    target.detach().cpu().numpy(),
                )

            trainer._train_batch = types.MethodType(prox_train_batch, trainer)

        try:
            client_agent.train()
        finally:
            trainer._train_batch = original_train_batch

        loss_after = self._evaluate_fixed_batches(trainer) if self.utility_enabled else None
        loss_delta = (
            loss_before - loss_after
            if loss_before is not None and loss_after is not None else None
        )
        finite_values = [x for x in (loss_before, loss_after, loss_delta) if x is not None]
        finite = all(math.isfinite(float(x)) for x in finite_values)
        batches = int(totals["batches"])
        return RUPTrainingObservation(
            client_id=client_id,
            local_steps=local_steps,
            num_train_samples=num_samples,
            loss_before=loss_before,
            loss_after=loss_after,
            loss_delta=loss_delta,
            loss_delta_per_step=(loss_delta / max(local_steps, 1)) if loss_delta is not None else None,
            prox_mu=self.prox_mu,
            mean_prox_penalty=totals["prox"] / max(batches, 1),
            mean_base_loss=totals["base"] / max(batches, 1),
            num_observed_batches=batches,
            finite=finite,
        )

    def _evaluate_fixed_batches(self, trainer) -> float:
        """Evaluate deterministic leading batches without advancing the shuffled loader."""
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        # Reuse the same augmentation realization before and after local training.
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        device = trainer.train_configs.device
        model = trainer.model
        was_training = model.training
        model.to(device)
        model.eval()
        loader = torch.utils.data.DataLoader(
            trainer.train_dataset,
            batch_size=trainer.train_configs.get("train_batch_size", 32),
            shuffle=False,
            num_workers=trainer.train_configs.get("num_workers", 0),
        )
        losses = []
        try:
            with torch.no_grad():
                for index, (data, target) in enumerate(loader):
                    if index >= self.utility_eval_batches:
                        break
                    output = model(data.to(device))
                    losses.append(float(trainer.loss_fn(output, target.to(device)).detach().cpu()))
        finally:
            if was_training:
                model.train()
            # Utility observation must not alter the stochastic training path.
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        return float(np.mean(losses)) if losses else float("nan")
