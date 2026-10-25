from collections.abc import Callable

import torch
from transformers import PreTrainedModel

from .decoding import CachedModel, Generation
from .trace import RoundEvent, SeedEvent


@torch.inference_mode()
def generate_speculative(
    target: PreTrainedModel,
    draft: PreTrainedModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    draft_tokens: int,
    eos_ids: frozenset[int],
    on_event: Callable[[SeedEvent | RoundEvent], None] | None = None,
) -> Generation:
    """Greedy draft/verify with one committed token kept outside both caches.

    At every round boundary:
        target_cache == draft_cache == committed_prefix_without_last_token
        pending == last committed token

    Agreement with ordinary greedy decoding is an algorithmic goal, subject to
    floating-point differences between block and single-token model calls.
    """
    target_state = CachedModel(target)
    pending = target_state.predict(prompt_ids)
    first_token = int(pending.item())
    generated = [first_token]
    if on_event is not None:
        on_event(SeedEvent(first_token, target_state.length))
    if first_token in eos_ids:
        return Generation(tuple(generated), "eos")
    if max_new_tokens == 1:
        return Generation(tuple(generated), "max_new_tokens")

    draft_state = CachedModel(draft)
    draft_state.predict(prompt_ids)
    round_index = 0

    while len(generated) < max_new_tokens:
        round_index += 1
        prefix_length = prompt_ids.shape[1] + len(generated)
        base_length = prefix_length - 1
        if target_state.length != base_length or draft_state.length != base_length:
            raise RuntimeError("Both caches must contain exactly the prefix before pending.")

        remaining = max_new_tokens - len(generated)
        proposal_limit = min(draft_tokens, remaining)
        proposal_tensors = []
        proposed = []
        draft_input = pending

        # The small model still drafts sequentially. Its final proposal is not
        # cached yet: predicting a token does not feed that token into the model.
        for _ in range(proposal_limit):
            proposal = draft_state.predict(draft_input)
            token_id = int(proposal.item())
            proposal_tensors.append(proposal)
            proposed.append(token_id)
            draft_input = proposal
            if token_id in eos_ids:
                break

        # For [pending, d1, d2, ...], the corresponding logits predict
        # [d1, d2, ..., bonus]. Causal attention prevents looking ahead.
        verification_input = torch.cat([pending, *proposal_tensors], dim=1)
        target_predictions = target_state.predict(
            verification_input, all_positions=True
        )[0].tolist()

        accepted = 0
        for proposal, prediction in zip(proposed, target_predictions):
            if proposal != prediction:
                break
            accepted += 1

        emitted = proposed[:accepted]
        target_token = None
        target_kind = None
        if accepted < len(proposed):
            target_token = target_predictions[accepted]
            target_kind = "correct"
        elif len(emitted) < remaining and emitted[-1] not in eos_ids:
            target_token = target_predictions[len(proposed)]
            target_kind = "bonus"
        if target_token is not None:
            emitted.append(target_token)

        # Drafting stops at draft EOS. An accepted EOS or a target correction
        # that is EOS therefore already sits at the end of emitted.
        generated.extend(emitted)
        keep_length = prompt_ids.shape[1] + len(generated) - 1

        # Discard the branch after the committed prefix. Leave its final token
        # pending, so the next verification call can predict the first proposal.
        target_state.crop(keep_length)
        if draft_state.length < keep_length:
            # Full acceptance + bonus: cache the final draft token too.
            # This consumes at most one token. The bonus itself remains pending.
            start = draft_state.length - base_length
            end = keep_length - base_length
            draft_state.predict(verification_input[:, start:end])
        draft_state.crop(keep_length)
        pending = torch.tensor([[generated[-1]]], dtype=prompt_ids.dtype, device=prompt_ids.device)

        if on_event is not None:
            on_event(
                RoundEvent(
                    index=round_index,
                    prefix_length=prefix_length,
                    cache_before=base_length,
                    cache_after=keep_length,
                    proposed=tuple(proposed),
                    accepted=accepted,
                    target_token=target_token,
                    target_kind=target_kind,
                    emitted=tuple(emitted),
                )
            )
        if generated[-1] in eos_ids:
            return Generation(tuple(generated), "eos")

    return Generation(tuple(generated), "max_new_tokens")
