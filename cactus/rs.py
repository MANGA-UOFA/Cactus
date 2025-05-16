# Copyright (c) 2026 The Cactus Authors

from typing import Dict, Optional, Tuple

import torch
import wandb
from vllm.logger import init_logger
from vllm.model_executor.layers.rejection_sampler import (
    RejectionSampler as OriginalRejectionSampler,
)
from vllm.model_executor.layers.rejection_sampler import _multinomial

from cactus.constant import tracker

logger = init_logger(__name__)


class WrappedRejectionSampler(OriginalRejectionSampler):
    real_time_al: bool = False

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
                target_with_bonus_probs, draft_token_ids, bonus_token_ids, draft_probs
            )

        batch_size, k, _ = draft_probs.shape

        if batch_size == 0:
            return torch.empty(0, k + 1, device=draft_probs.device, dtype=torch.int)

        accepted, recovered_token_ids = self._batch_modified_rejection_sampling(
            target_with_bonus_probs[:, :-1],
            draft_probs,
            draft_token_ids,
            seeded_seqs,
        )
        output_token_ids = self._create_output(
            accepted,
            recovered_token_ids,
            draft_token_ids,
            bonus_token_ids,
        )

        al = accepted.cumprod(dim=-1).float().sum(dim=-1).mean().item()
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

    def _batch_modified_rejection_sampling(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        seeded_seqs: Optional[Dict[int, torch.Generator]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform modified rejection sampling on each sequence.

        Returns:
            A tuple of two tensors:
            0: A bool tensor of which tokens in each sequence is accepted.
                shape = [batch_size, k]
            1: Token ids sampled from a recovered distribution, to be used
                when a token is rejected.
                shape = [batch_size, k]
        """

        batch_size, k, vocab_size = draft_probs.shape

        # shape [batch_size, k]
        accepted = self._get_accepted(target_probs, draft_probs, draft_token_ids, seeded_seqs)

        recovered_probs = self._get_recovered_probs(
            target_probs,
            draft_probs,
            draft_token_ids,
        ).reshape(batch_size * k, vocab_size)

        # NOTE: the recovered_probs are overwritten by this method.
        recovered_token_ids = _multinomial(
            recovered_probs,
            num_samples=1,
            k=k,
            seeded_seqs=seeded_seqs or {},
        ).reshape(batch_size, k)

        return accepted, recovered_token_ids

    def _get_recovered_probs(  # type: ignore[override]
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
    ) -> torch.Tensor:
        return super(WrappedRejectionSampler, self)._get_recovered_probs(target_probs, draft_probs)

    def _get_accepted(
        self,
        target_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_probs: torch.Tensor,  # [batch_size, k, vocab_size]
        draft_token_ids: torch.Tensor,  # [batch_size, k]
        seeded_seqs: Optional[Dict[int, torch.Generator]],
    ) -> torch.Tensor:
        return super()._get_accepted(target_probs, draft_probs, draft_token_ids, seeded_seqs)
