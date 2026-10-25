from dataclasses import dataclass
import json
import sys
from typing import TextIO, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class SeedEvent:
    token_id: int
    cache_length: int


@dataclass(frozen=True)
class RoundEvent:
    index: int
    prefix_length: int
    cache_before: int
    cache_after: int
    proposed: tuple[int, ...]
    accepted: int
    target_token: int | None
    target_kind: str | None
    emitted: tuple[int, ...]


class TracePrinter:
    def __init__(self, tokenizer: "PreTrainedTokenizerBase", stream: TextIO | None = None):
        self.tokenizer = tokenizer
        self.stream = sys.stderr if stream is None else stream

    def token(self, token_id: int) -> str:
        piece = self.tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        # Escaping also keeps newlines and terminal control characters inside the label.
        return f"{token_id}:{json.dumps(piece, ensure_ascii=True)}"

    def __call__(self, event: SeedEvent | RoundEvent) -> None:
        if isinstance(event, SeedEvent):
            print(
                f"seed: {self.token(event.token_id)} | cached_prompt={event.cache_length}",
                file=self.stream,
            )
            return
        print(
            f"round {event.index} | prefix={event.prefix_length} | "
            f"both_caches={event.cache_before}->{event.cache_after}",
            file=self.stream,
        )
        for position, token_id in enumerate(event.proposed):
            if position < event.accepted:
                status = "accept"
            elif position == event.accepted:
                status = "reject"
            else:
                status = "discard"
            print(f"  {status:7} {self.token(token_id)}", file=self.stream)
        if event.target_token is not None:
            print(
                f"  {event.target_kind:7} {self.token(event.target_token)}",
                file=self.stream,
            )
        emitted = " ".join(self.token(token_id) for token_id in event.emitted)
        print(f"  emitted {emitted}", file=self.stream)

