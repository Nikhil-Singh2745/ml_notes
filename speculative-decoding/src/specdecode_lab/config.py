from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    mode: str
    target: str
    draft: str
    target_revision: str
    draft_revision: str
    device: str
    dtype: str
    attention: str
    max_new_tokens: int
    max_context: int
    draft_tokens: int
    local_files_only: bool

    def validate(self) -> None:
        if self.mode not in {"baseline", "speculative"}:
            raise ValueError("mode must be baseline or speculative")
        if self.max_new_tokens < 1:
            raise ValueError("max-new-tokens must be positive")
        if self.max_context < 2:
            raise ValueError("max-context must be at least 2")
        if self.max_new_tokens >= self.max_context:
            raise ValueError("max-new-tokens must leave room for the prompt")
        if not 1 <= self.draft_tokens <= 32:
            raise ValueError("draft-tokens must be between 1 and 32")
        if not (
            self.device == "cuda"
            or (
                self.device.startswith("cuda:")
                and self.device[5:].isascii()
                and self.device[5:].isdigit()
            )
        ):
            raise ValueError("device must be cuda or cuda:N; there is no CPU fallback")

