# Copyright (c) 2026 The Cactus Authors
#
# Triton kernel for chain speculative sampling with Cactus transform.
# Cactus boosts draft token acceptance by transforming
#   target probs q → h, where h_n = clamp(q_n + sqrt(2·delta·q_n·(1-q_n)), 0, 1).
#   The transform is computed on-the-fly — the [B, k, V] tensor h is never materialized.


from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=32),
    ],
    key=["vocab_size", "batch_bucket"],
)
@triton.jit
def _kernel(
    draft_probs_ptr,
    draft_token_ids_ptr,
    target_probs_ptr,
    bonus_token_ids_ptr,
    output_ids_ptr,
    accepted_num_ptr,
    seed,
    num_spec_tokens,
    vocab_size,
    delta,
    batch_bucket,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    kp1 = num_spec_tokens + 1

    # Phase 1: determine acceptance prefix
    pos = num_spec_tokens
    h_n_at_pos = 0.0  # float32: h_n
    scale_at_pos = 1.0  # float32: precomputed (1-h_n)/(1-q_n)

    i = 0
    while i < num_spec_tokens and pos == num_spec_tokens:
        draft_id = tl.load(draft_token_ids_ptr + row * num_spec_tokens + i)
        q_n = tl.load(target_probs_ptr + (row * kp1 + i) * vocab_size + draft_id)
        p_n = tl.load(draft_probs_ptr + (row * num_spec_tokens + i) * vocab_size + draft_id)

        # obtain h_n following Cactus' theory
        h_n = q_n + tl.sqrt(2.0 * delta * q_n * (1.0 - q_n))
        h_n = tl.minimum(tl.maximum(h_n, 0.0), 1.0)

        if tl.rand(seed, row * num_spec_tokens + i) * p_n < h_n:
            tl.store(output_ids_ptr + row * kp1 + i, draft_id)
        else:
            pos = i
            h_n_at_pos = h_n
            scale_at_pos = (1.0 - h_n) / tl.maximum(1.0 - q_n, 1e-10)
        i += 1

    tl.store(accepted_num_ptr + row, pos)  # accepted = emitted for early-exit

    # Phase 2: recovery sampling from weight max(0, h - p), or bonus token store if accepted all
    if pos == num_spec_tokens:
        bonus_id = tl.load(bonus_token_ids_ptr + row)
        tl.store(output_ids_ptr + row * kp1 + num_spec_tokens, bonus_id)
    else:
        target_off = (row * kp1 + pos) * vocab_size
        draft_off = (row * num_spec_tokens + pos) * vocab_size
        draft_id_at_pos = tl.load(draft_token_ids_ptr + row * num_spec_tokens + pos)

        # One-pass reservoir sampling from max(0, h - p)
        # For each chunk: with prob chunk_sum / (W + chunk_sum), replace the
        # current sample with one drawn from this chunk via inverse CDF.
        # After all chunks: P(token j) = w_j / Z.
        W = 0.0
        sampled_id = 0

        chunk_start = 0
        while chunk_start < vocab_size:
            offsets = chunk_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < vocab_size
            q = tl.load(target_probs_ptr + target_off + offsets, mask=mask, other=0.0)
            p = tl.load(draft_probs_ptr + draft_off + offsets, mask=mask, other=0.0)
            h = tl.where(offsets == draft_id_at_pos, h_n_at_pos, q * scale_at_pos)
            relu_diff = tl.maximum(h - p, 0.0)

            chunk_sum = tl.sum(relu_diff)
            W_new = W + chunk_sum

            # Reservoir: replace sample with prob chunk_sum / W_new
            if tl.rand(seed + 1, row * vocab_size + chunk_start) * W_new < chunk_sum:
                # Inverse CDF within this chunk
                # Scale by cumsum's total (not tl.sum's) so a crossing is guaranteed.
                valid = (relu_diff > 0) & mask
                cumsum = tl.cumsum(relu_diff, axis=0)
                u_inner = tl.rand(seed + 2, row * vocab_size + chunk_start) * tl.max(
                    tl.where(valid, cumsum, 0.0)
                )
                sampled_id = tl.min(tl.where((cumsum > u_inner) & valid, offsets, vocab_size))

            W = W_new
            chunk_start += BLOCK_SIZE

        tl.store(output_ids_ptr + row * kp1 + pos, sampled_id)

    # Phase 3: pad remaining with -1
    j = pos + 1
    while j < kp1:
        tl.store(output_ids_ptr + row * kp1 + j, -1)
        j += 1


def fused_cactus(
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    delta: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chain speculative sampling with optional Cactus transform.

    Args:
        draft_probs:     [batch_size, num_speculate_tokens, vocab_size]
        draft_token_ids: [batch_size, num_speculate_tokens]
        target_probs:    [batch_size, num_speculate_tokens + 1, vocab_size]
        bonus_token_ids: [batch_size, 1]
        delta:           Cactus divergence parameter.

    Returns:
        output:          [batch_size, num_speculate_tokens + 1] - accepted draft tokens
                         followed by one recovery token
        accepted_num:    [batch_size] - how many draft tokens were accepted
    """
    batch_size, num_spec_tokens, vocab_size = draft_probs.shape
    device = draft_probs.device

    output = torch.empty((batch_size, num_spec_tokens + 1), dtype=torch.int64, device=device)
    accepted_num = torch.zeros(batch_size, dtype=torch.int64, device=device)

    if batch_size == 0:
        return output, accepted_num

    _kernel[(batch_size,)](
        draft_probs.contiguous(),
        draft_token_ids.contiguous(),
        target_probs.contiguous(),
        bonus_token_ids.contiguous(),
        output,
        accepted_num,
        int(torch.randint(0, 2**31, (1,)).item()),
        num_spec_tokens,
        vocab_size,
        delta,
        1 + batch_size // 4,
    )
    return output, accepted_num
