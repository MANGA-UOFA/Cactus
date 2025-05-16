# Copyright (c) 2026 The Cactus Authors

from typing import Dict, Optional

import torch
import torch.jit
from vllm.logger import init_logger

from cactus.rs import WrappedRejectionSampler

logger = init_logger(__name__)


class InterpRejectionSampler(WrappedRejectionSampler):
    """Interpolation-based acceptance sampler.

    Uses an interpolated distribution between draft and target:
        interp = alpha * draft + (1 - alpha) * target

    alpha = 0 recovers standard rejection sampling (pure target).
    alpha = 1 accepts everything (pure draft).
    """

    alpha: float = 0.5

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

        alpha = self.alpha

        interp_probs = alpha * draft_probs + (1 - alpha) * target_probs

        selected_draft_probs = draft_probs[batch_indices, probs_indices, draft_token_ids]
        selected_interp_probs = interp_probs[batch_indices, probs_indices, draft_token_ids]

        uniform_rand = self._create_uniform_samples(
            seeded_seqs, batch_size, k - 1, target_probs.device
        )

        capped_ratio = torch.minimum(
            selected_interp_probs / selected_draft_probs,
            torch.full((1,), 1, device=target_probs.device),
        )

        accepted = uniform_rand < capped_ratio

        return accepted

    def _get_recovered_probs(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
    ) -> torch.Tensor:
        alpha = self.alpha

        interp_probs = alpha * draft_probs + (1 - alpha) * target_probs

        difference = interp_probs - draft_probs
        f = torch.clamp(difference, min=self._smallest_positive_value)
        recovered_probs = f / torch.sum(f, dim=-1, keepdim=True)

        return recovered_probs
