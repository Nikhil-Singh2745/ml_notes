import argparse
from pathlib import Path
import sys

from .config import RunConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Greedy local decoding with dense Qwen3 models.")
    parser.add_argument("--mode", choices=("baseline", "speculative"), default="speculative")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="One user message.")
    source.add_argument("--prompt-file", type=Path, help="UTF-8 file containing one user message.")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--target", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--target-revision", default="main", help="Hub revision or commit hash.")
    parser.add_argument("--draft-revision", default="main", help="Hub revision or commit hash.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=4096, help="Prompt + output token budget.")
    parser.add_argument("--draft-tokens", type=int, default=4, help="Proposals per round, from 1 to 32.")
    parser.add_argument("--local-files-only", action="store_true", help="Use already cached artifacts.")
    parser.add_argument("--trace", action="store_true", help="Print draft decisions to stderr.")
    parser.add_argument("--debug", action="store_true", help="Keep the full Python error traceback.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.trace and args.mode != "speculative":
        parser.error("--trace is only available in speculative mode")
    fields = RunConfig.__dataclass_fields__
    config = RunConfig(**{key: getattr(args, key) for key in fields})

    try:
        config.validate()
        prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
        if not prompt.strip():
            raise ValueError("The prompt cannot be empty.")

        # Parse arguments before importing the model stack. --help stays lightweight.
        from .decoding import generate_baseline
        from .models import load_models, prepare_prompt

        models = load_models(config)
        prompt_ids = prepare_prompt(models, prompt, args.system, config.max_new_tokens)
        if config.mode == "baseline":
            result = generate_baseline(
                models.target, prompt_ids, config.max_new_tokens, models.eos_ids
            )
        else:
            from .speculative import generate_speculative
            from .trace import TracePrinter

            result = generate_speculative(
                target=models.target,
                draft=models.draft,
                prompt_ids=prompt_ids,
                max_new_tokens=config.max_new_tokens,
                draft_tokens=config.draft_tokens,
                eos_ids=models.eos_ids,
                on_event=TracePrinter(models.tokenizer) if args.trace else None,
            )
        text = models.tokenizer.decode(
            list(result.token_ids), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(text)
        print(f"stop={result.stop_reason} generated_tokens={len(result.token_ids)}", file=sys.stderr)
        return 0
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        if args.debug:
            raise
        print(f"error: {error}", file=sys.stderr)
        print("Use --debug for the full traceback.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
