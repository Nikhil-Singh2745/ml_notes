import sys
from dataclasses import dataclass

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import RunConfig


@dataclass
class Models:
    target: PreTrainedModel
    draft: PreTrainedModel | None
    tokenizer: PreTrainedTokenizerBase
    eos_ids: frozenset[int]
    context_limit: int


def load_models(config: RunConfig) -> Models:
    """Check CUDA before fetching anything. Keep both models entirely on one GPU."""
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this on the NVIDIA desktop.")

    device = torch.device(config.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support bfloat16; try --dtype float16.")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
    print(
        f"GPU: {torch.cuda.get_device_name(device)} | "
        f"torch {torch.__version__} | CUDA {torch.version.cuda} | "
        f"transformers {transformers.__version__}",
        file=sys.stderr,
    )
    print(f"mode={config.mode} dtype={config.dtype} attention={config.attention}", file=sys.stderr)

    common = {"trust_remote_code": False, "local_files_only": config.local_files_only}
    target_config = AutoConfig.from_pretrained(
        config.target, revision=config.target_revision, **common
    )
    _check_architecture(target_config)
    # Resolve main once, then use the resolved revision for weights and tokenizer.
    target_revision = getattr(target_config, "_commit_hash", None) or config.target_revision
    tokenizer = AutoTokenizer.from_pretrained(
        config.target, revision=target_revision, **common
    )

    draft_config = None
    draft_revision = config.draft_revision
    if config.mode == "speculative":
        draft_config = AutoConfig.from_pretrained(
            config.draft, revision=config.draft_revision, **common
        )
        _check_architecture(draft_config)
        draft_revision = getattr(draft_config, "_commit_hash", None) or config.draft_revision
        draft_tokenizer = AutoTokenizer.from_pretrained(
            config.draft, revision=draft_revision, **common
        )
        if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
            raise ValueError("The target and draft must have identical token-to-ID mappings.")
        if tokenizer.all_special_ids != draft_tokenizer.all_special_ids:
            raise ValueError("The target and draft special-token IDs must match.")
        if target_config.vocab_size != draft_config.vocab_size:
            raise ValueError("The target and draft logit vocabulary sizes must match.")

    context_limit = min(config.max_context, target_config.max_position_embeddings)
    if draft_config is not None:
        context_limit = min(context_limit, draft_config.max_position_embeddings)

    load_options = {
        **common,
        "dtype": dtype,
        "device_map": {"": str(device)},
        "attn_implementation": config.attention,
        "use_safetensors": True,
    }
    print(f"target: {config.target} @ {target_revision}", file=sys.stderr)
    target = AutoModelForCausalLM.from_pretrained(
        config.target, revision=target_revision, config=target_config, **load_options
    ).eval()
    draft = None
    if draft_config is not None:
        print(f"draft:  {config.draft} @ {draft_revision}", file=sys.stderr)
        draft = AutoModelForCausalLM.from_pretrained(
            config.draft, revision=draft_revision, config=draft_config, **load_options
        ).eval()

    eos = target.generation_config.eos_token_id
    if eos is None:
        eos = target.config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = frozenset(eos if isinstance(eos, list) else [eos]) if eos is not None else frozenset()
    return Models(target, draft, tokenizer, eos_ids, context_limit)


def _check_architecture(config) -> None:
    if config.model_type != "qwen3":
        raise ValueError("This implementation is scoped to dense Qwen3 models.")
    if getattr(config, "use_sliding_window", False):
        raise ValueError("Sliding-window caches are outside this implementation's scope.")


def prepare_prompt(models: Models, prompt: str, system: str, max_new_tokens: int) -> torch.Tensor:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    text = models.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    # The chat template already inserted its special tokens.
    encoded = models.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids.shape[1] + max_new_tokens > models.context_limit:
        raise ValueError(
            f"Prompt ({ids.shape[1]}) + requested output ({max_new_tokens}) exceeds "
            f"the context budget ({models.context_limit}). Shorten one or raise --max-context. "
            "The prompt is never silently truncated."
        )
    print(f"prompt_tokens={ids.shape[1]} context_budget={models.context_limit}", file=sys.stderr)
    return ids.to(models.target.device)

