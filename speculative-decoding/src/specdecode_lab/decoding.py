from dataclasses import dataclass

import torch
from transformers import DynamicCache, PreTrainedModel


@dataclass(frozen=True)
class Generation:
    token_ids: tuple[int, ...]
    stop_reason: str


class CachedModel:
    """Each forward appends only previously uncached input tokens."""

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.cache = DynamicCache(config=model.config)

    @property
    def length(self) -> int:
        return self.cache.get_seq_length()

    def predict(self, ids: torch.Tensor, *, all_positions: bool = False) -> torch.Tensor:
        positions = torch.arange(self.length, self.length + ids.shape[1], device=ids.device)
        output = self.model(
            input_ids=ids,
            past_key_values=self.cache,
            cache_position=positions,
            position_ids=positions.unsqueeze(0),
            use_cache=True,
            return_dict=True,
            # Verification needs a prediction at each draft position.
            # Prefill and ordinary single-token steps only need the last one.
            logits_to_keep=0 if all_positions else 1,
        )
        self.cache = output.past_key_values
        return output.logits.argmax(dim=-1)

    def crop(self, length: int) -> None:
        if not 0 <= length <= self.length:
            raise RuntimeError(f"Cannot crop cache of length {self.length} to {length}.")
        self.cache.crop(length)


@torch.inference_mode()
def generate_baseline(
    model: PreTrainedModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_ids: frozenset[int],
) -> Generation:
    state = CachedModel(model)
    next_id = state.predict(prompt_ids)
    generated = []
    for index in range(max_new_tokens):
        token_id = int(next_id.item())
        generated.append(token_id)
        if token_id in eos_ids:
            return Generation(tuple(generated), "eos")
        if index + 1 < max_new_tokens:
            next_id = state.predict(next_id)
    return Generation(tuple(generated), "max_new_tokens")
