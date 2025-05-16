# Copyright (c) 2026 The Cactus Authors

import torch
import wandb
from vllm.model_executor.layers.typical_acceptance_sampler import (
    TypicalAcceptanceSampler as OriginalTypicalAcceptanceSampler,
)

from cactus.constant import tracker


class WrappedTypicalAcceptanceSampler(OriginalTypicalAcceptanceSampler):
    def forward(
        self,
        target_with_bonus_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sample token ids using typical acceptance sampling. This accepts
        or rejects tokens proposed by the draft model using the probability
        of each token according to the draft and target models.

        In the worst case where all draft tokens are rejected, it is guaranteed
        one token will be emitted.

        In the case where all draft tokens are accepted, the bonus token will be
        accepted.

        Args:
            target_probs: The probability distribution over token ids given
                context according to the target model.
            shape = [batch_size, num_speculative_tokens, vocab_size]

            bonus_token_ids: The "bonus" token ids that are accepted iff all
                speculative tokens in a sequence are accepted.
            shape = [batch_size, num_bonus_tokens]

            draft_probs: This parameter is unused by the acceptance sampler.

            draft_token_ids: The token ids that were sampled from the draft
                probabilities.
            shape = [batch_size, num_speculative_tokens]

        Returns:
            output_token_ids: The token ids sampled via rejection sampling,
                or -1 if unable to sample a token because the previous token
                was rejected.
            shape = [batch_size, num_speculative_tokens + num_bonus_tokens]
        """
        # Only perform shape/dtype/device checking in strict mode, as it adds
        # overhead.
        if self._strict_mode:
            self._raise_if_incorrect_input(
                target_with_bonus_probs, draft_token_ids, bonus_token_ids
            )

        batch_size, k, _ = draft_probs.shape
        if batch_size == 0:
            return torch.empty(0, k + 1, device=draft_probs.device, dtype=torch.int)

        target_probs = target_with_bonus_probs[:, :-1]
        accepted = self._evaluate_accepted_tokens(target_probs, draft_token_ids)
        recovered_token_ids = self._get_recovered_token_ids(target_probs)
        output_token_ids = self._create_output(
            accepted, recovered_token_ids, draft_token_ids, bonus_token_ids
        )

        # Custom logging
        running_task = tracker.task
        k = accepted.shape[1]
        al = accepted.float().cumprod(dim=-1).sum(dim=-1).mean().item()
        avg_al = tracker.update(al)

        wandb.log(
            {
                f"{running_task}/accepted_length": al,
                f"{running_task}/avg_accepted_length": avg_al,
            }
        )

        return output_token_ids
