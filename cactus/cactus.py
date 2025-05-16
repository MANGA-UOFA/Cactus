# Copyright (c) 2026 The Cactus Authors

from typing import Dict, Optional

import torch
import torch.jit
import wandb
from vllm.logger import init_logger

from cactus.constant import tracker
from cactus.rs import WrappedRejectionSampler as WrappedRejectionSampler

logger = init_logger(__name__)

try:
    from cactus.triton import fused_cactus as triton_fc
except (ImportError, ModuleNotFoundError):
    triton_fc = None


def solve_constraint(
    target_probs: torch.Tensor,  # q,  shape [B, L, V]
    draft_probs: torch.Tensor,  # p,  shape [B, L, V] (only p_n used)
    selected_indices: torch.Tensor,  # n,  shape [B, L]
    delta: float,
    selected_only: bool = False,
):
    index = selected_indices.unsqueeze(-1)  # [B, L, 1]
    q_n = target_probs.gather(-1, index).squeeze(-1)  # [B, L]
    h_n = q_n + torch.sqrt(2 * delta * q_n * (1 - q_n))
    h_n = h_n.clamp(min=0.0, max=1.0)  # [B, L]
    if selected_only:
        return h_n
    h = target_probs * ((1 - h_n) / (1 - q_n)).nan_to_num(
        nan=1.0, posinf=1.0, neginf=0.0
    ).unsqueeze(-1)  # [B, L, V]
    h.scatter_(-1, index, h_n.unsqueeze(-1))  # [B, L, V]
    h = h.clamp(min=0.0, max=1.0)  # [B, L, V]
    return h  # [B, L, V]


class CactusRejectionSampler(WrappedRejectionSampler):
    delta: float
    enable_experimental_triton: bool = False

    def forward(
        self,
        target_with_bonus_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        seeded_seqs: Optional[Dict[int, torch.Generator]] = None,
    ) -> torch.Tensor:
        if self._strict_mode:
            self._raise_if_incorrect_input(
                target_with_bonus_probs,
                draft_token_ids,
                bonus_token_ids,
                draft_probs,
            )

        batch_size, k, _ = draft_probs.shape

        if batch_size == 0:
            return torch.empty(0, k + 1, device=draft_probs.device, dtype=torch.int)

        if triton_fc is not None and self.enable_experimental_triton:
            logger.warning_once("Using experimental Triton-optimized Cactus sampling.")
            output_token_ids, accepted_token_num = triton_fc(
                draft_probs,
                draft_token_ids,
                target_with_bonus_probs,
                bonus_token_ids,
                delta=self.delta,
            )

            self.num_accepted_tokens += accepted_token_num.sum()
            self.num_emitted_tokens += accepted_token_num.sum() + batch_size
            self.num_draft_tokens += batch_size * k

            al = accepted_token_num.float().mean().item()
            tracker.update(al)

            if self.real_time_al:
                running_task = tracker.task
                wandb.log(
                    {
                        f"{running_task}/accepted_length": al,
                        f"{running_task}/avg_accepted_length": tracker.avg,
                    }
                )

            return output_token_ids

        # Triton unavailable — fall back to PyTorch path.
        # super().forward() dispatches to _get_accepted / _get_recovered_probs
        # which are overridden below with delta support.
        return super().forward(
            target_with_bonus_probs,
            bonus_token_ids,
            draft_probs,
            draft_token_ids,
            seeded_seqs,
        )

    def _get_accepted(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        seeded_seqs: Optional[Dict[int, torch.Generator]],
    ) -> torch.Tensor:
        batch_size, k, _ = draft_probs.shape
        batch_indices = torch.arange(batch_size, device=target_probs.device)[:, None]
        probs_indices = torch.arange(k, device=target_probs.device)

        # shape [batch_size, k]
        selected_draft_probs = draft_probs[batch_indices, probs_indices, draft_token_ids]

        uniform_rand = self._create_uniform_samples(
            seeded_seqs, batch_size, k - 1, target_probs.device
        )

        # CUSTOMIZED
        delta = self.delta
        h_n = solve_constraint(
            target_probs,
            draft_probs,
            draft_token_ids,
            delta,
            selected_only=True,
        )

        capped_ratio = torch.minimum(
            h_n / selected_draft_probs, torch.full((1,), 1, device=target_probs.device)
        )
        # End CUSTOMIZED

        accepted = uniform_rand < capped_ratio

        return accepted

    def _get_recovered_probs(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
    ) -> torch.Tensor:
        # CUSTOMIZED
        delta = self.delta

        h = solve_constraint(target_probs, draft_probs, draft_token_ids, delta)

        h_minus_p_phi = torch.clamp(h - draft_probs, min=0.0)

        recovered_probs = h_minus_p_phi / h_minus_p_phi.sum(dim=-1, keepdim=True)
        # prevent NaN
        recovered_probs = torch.nan_to_num(recovered_probs, nan=0.0, posinf=1.0, neginf=0.0)

        return recovered_probs
