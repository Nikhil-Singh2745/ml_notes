# speculative decoding, one GPU

I'm working through how a small model can help a bigger one generate text. This
repo keeps the decoding loop visible, with notes on the parts I want to understand
well enough to explain: token positions, the first rejected proposal, and keeping
two attention caches on the same branch.

The small model proposes a few tokens. The bigger model checks them in one forward
pass. Keep the matching prefix, correct the first mismatch, and continue. Drafting
adds work too; whether it pays off depends on how many proposals the target accepts
and the cost of running both models.

I'm starting with Qwen3-8B as the target and Qwen3-0.6B as the draft, both on one
GPU. Transformers handles the model internals. The ordinary greedy loop and the
speculative loop live here so you can follow them, run them, and change them.

## What you need

My home PC is a Ryzen 9950X3D with 96 GB RAM and an RTX 5090, running Fedora KDE
Plasma Desktop 44. That's the machine I'm setting this up for. To try it yourself,
the main constraint is enough GPU memory for your chosen model pair and context.

This implementation uses an NVIDIA CUDA GPU and keeps both models entirely in
VRAM. The default BF16 pair has roughly 16.4 GiB of weights alone, with caches and
temporary allocations on top. System RAM is a separate budget; the loader doesn't
offload weights to it. The [GPU note](notes/05-fitting-this-on-the-gpu.md) walks
through the memory arithmetic and options for a smaller card.

You'll also need Python 3.11–3.13, a working NVIDIA driver compatible with CUDA
12.8, and disk space for the environment and model downloads. I'm using Python
3.12 in the commands below. These are Linux shell commands; on another Linux
distribution, install Python and its venv support through your usual package
manager first.

## Set up the environment

First, check that the driver can see your GPU:

```bash
nvidia-smi
```

On Fedora, install the versioned
[Python 3.12 package](https://packages.fedoraproject.org/pkgs/python3.12/python3.12/):

```bash
sudo dnf install python3.12
```

From the repo directory, create a virtual environment and install the project:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

Run the `pip` commands inside the activated environment, without `sudo`. Using a
versioned interpreter also avoids picking a system Python outside the project's
supported range.

I'm pinning the direct runtime dependencies in [pyproject.toml](pyproject.toml),
including Transformers because the loop uses its cache API directly. PyTorch's
[versioned install instructions](https://pytorch.org/get-started/previous-versions/#v291)
list the CUDA 12.8 wheel. This build also covers the 5090's Blackwell architecture;
PyTorch's [Blackwell support announcement](https://pytorch.org/blog/pytorch-2-7/)
has the background. The wheel supplies its CUDA runtime, so this Python-only
project doesn't need a separate CUDA toolkit installation.

To check the entry point without loading models:

```bash
specdecode --help
```

`python -m specdecode_lab` also works after installation. Activate `.venv` again
when opening a new shell.

## Start with ordinary generation

Run the target on its own first. This is the reference loop for the speculative
version:

```bash
specdecode --mode baseline \
  --prompt "Write a short Python function that merges two sorted lists."
```

The first model run downloads the selected weights and tokenizer files from
Hugging Face. Allow roughly 20 GB of disk space for the default pair's weights,
plus the Python environment and download overhead. Downloads go into the normal
Hugging Face cache. Set `HF_HOME` before running if you want that cache elsewhere;
`--local-files-only` uses files already cached.

The final continuation prints once generation finishes. The prompt isn't echoed.
Startup details and the stop reason go to stderr, while the generated text goes
to stdout.

## Try speculative generation

Use the same prompt, this time loading the draft as well:

```bash
specdecode --mode speculative \
  --prompt "Write a short Python function that merges two sorted lists." \
  --draft-tokens 4
```

Both loops use greedy decoding with thinking mode disabled. Keep the prompt,
system message, target revision, dtype, attention setting, and output limit the
same when comparing them. The [implementation choices](notes/04-choices-in-this-version.md)
explain the relationship between the two loops, including floating-point effects
on near-tied predictions.

For a longer prompt, use `--prompt-file path/to/prompt.txt`. The file supplies one
UTF-8 user message. `--system` changes the system message.

### Follow a round in the trace

For a closer look at what the draft proposes and the target keeps:

```bash
specdecode --mode speculative \
  --prompt "Explain why a Python dictionary lookup is usually fast." \
  --max-new-tokens 96 \
  --trace
```

The trace goes to stderr. It starts with `seed`, the target's first generated
token, then shows each round's decisions:

| Label | Meaning |
| --- | --- |
| `accept` | A proposal matches the target's greedy prediction |
| `reject` | The first proposal that disagrees with the target |
| `discard` | A later proposal on the rejected branch |
| `correct` | The target's replacement at the first mismatch |
| `bonus` | One extra target token after all proposals match |
| `both_caches` | Logical cache lengths before and after reconciliation |

Read the token IDs first. Individual decoded pieces are labels; some characters
span several tokens. The trace escapes whitespace and non-ASCII text. Turn it off
when you only want generation, since decoding labels and printing add work.

## Adjust the run

| Flag | Default | Meaning |
| --- | --- | --- |
| `--target` | `Qwen/Qwen3-8B` | Model whose predictions decide the output |
| `--draft` | `Qwen/Qwen3-0.6B` | Model proposing tokens in speculative mode |
| `--draft-tokens` | `4` | Maximum proposals per round; allowed range 1–32 |
| `--max-new-tokens` | `128` | Output token cap, including a generated EOS |
| `--max-context` | `4096` | Prompt plus requested output budget |
| `--device` | `cuda:0` | GPU holding the whole model pair |
| `--dtype` | `bfloat16` | Weight/compute dtype; also accepts `float16` |
| `--attention` | `sdpa` | Attention implementation; also accepts `eager` |
| `--target-revision` | `main` | Target Hub revision or commit hash |
| `--draft-revision` | `main` | Draft Hub revision or commit hash |
| `--debug` | off | Show the full traceback on errors |

The loader prints resolved model revisions to stderr. Copy those commit hashes
into the revision flags to reuse the same artifacts across runs.

For less VRAM, try a smaller target while keeping the same draft:

```bash
specdecode --mode speculative \
  --target Qwen/Qwen3-1.7B \
  --draft Qwen/Qwen3-0.6B \
  --prompt "Write a short Python function that merges two sorted lists." \
  --max-context 2048 \
  --max-new-tokens 64
```

Use that same `--target` in baseline mode when comparing the two loops. Model
choices must be dense, full-attention Qwen3 checkpoints with matching token-to-ID
mappings, special-token IDs, and logit vocabulary sizes. The loader checks these
before loading both weight sets. Each pair still needs room for its caches and
temporary allocations.

## If something gets in the way

| Symptom | What to check or change |
| --- | --- |
| `specdecode` isn't found | Activate `.venv` and run `python -m pip install -e .` from the repo directory |
| CUDA is unavailable | Check `nvidia-smi`, the active environment, and that PyTorch is a CUDA build |
| BF16 isn't supported | Try `--dtype float16`; both formats use two bytes per weight |
| CUDA runs out of memory | Close other GPU applications, then follow the [memory guide](notes/05-fitting-this-on-the-gpu.md#if-memory-runs-short) |
| Prompt exceeds the context budget | Shorten the prompt or output request, or raise `--max-context` if VRAM allows |
| Cached model files are missing | Remove `--local-files-only` to allow the download |

The context budget includes chat-template tokens and is capped by the loaded
model configs. An oversized prompt is rejected rather than truncated. Add
`--debug` to a failing command for the full traceback.

## Read alongside the code

The notes follow the order I use to reason about the loop:

| Note | What I'm keeping there |
| --- | --- |
| [01 — tokens first](notes/01-tokens-first.md) | Inputs and ordinary generation |
| [02 — checking a draft](notes/02-checking-a-draft.md) | Verification positions and the first mismatch |
| [03 — keeping the caches honest](notes/03-keeping-the-caches-honest.md) | The pending token and cache rollback |
| [04 — choices in this version](notes/04-choices-in-this-version.md) | Greedy decoding, model compatibility, and implementation costs |
| [05 — fitting this on the GPU](notes/05-fitting-this-on-the-gpu.md) | Memory arithmetic and adapting the run to your card |

The corresponding code is in:

| File | What it does |
| --- | --- |
| [decoding.py](src/specdecode_lab/decoding.py) | Cached model calls and ordinary greedy generation |
| [speculative.py](src/specdecode_lab/speculative.py) | Draft, verify, commit, and reconcile caches |
| [models.py](src/specdecode_lab/models.py) | CUDA loading, model checks, and prompt template |
| [trace.py](src/specdecode_lab/trace.py) | Token decision display |
| [cli.py](src/specdecode_lab/cli.py) | Command-line entry point |

The draft/parallel-verification idea comes from
[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192).
I'm implementing greedy token matching here. The paper's probability-based
acceptance/rejection sampler for temperature sampling is a separate algorithm.

Model references: [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B),
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B),
[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B).
