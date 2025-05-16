# Copyright (c) 2026 The Cactus Authors

from typing import Dict, Optional

import torch
import torch.jit
from vllm.logger import init_logger

from cactus.rs import WrappedRejectionSampler as WrappedRejectionSampler

logger = init_logger(__name__)


class TopKRejectionSampler(WrappedRejectionSampler):
    topk: int = 5

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

        selected_target_probs = target_probs[batch_indices, probs_indices, draft_token_ids]
        topk = self.topk

        threshold = target_probs.topk(topk, dim=-1, largest=True, sorted=True).values[..., topk - 1]
        accepted = selected_target_probs >= threshold

        return accepted

    def _get_recovered_probs(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
    ) -> torch.Tensor:
        batch_size, k, vocab_size = draft_probs.shape

        topk = self.topk

        threshold = (
            target_probs.topk(topk, dim=-1, largest=True, sorted=True)
            .values[..., topk - 1]
            .unsqueeze(-1)
        )
        h = (target_probs >= threshold).float()

        h_minus_p_phi = torch.clamp(h - draft_probs, min=0.0)
        recovered_probs = h_minus_p_phi / h_minus_p_phi.sum(dim=-1, keepdim=True)

        recovered_probs = torch.nan_to_num(recovered_probs, nan=0.0, posinf=1.0, neginf=0.0)

        return recovered_probs
