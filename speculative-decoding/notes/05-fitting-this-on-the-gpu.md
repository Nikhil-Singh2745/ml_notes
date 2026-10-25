# fitting this on the GPU

My home PC has a Ryzen 9950X3D, 96 GB RAM, and an RTX 5090. The
[desktop 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
has 32 GB VRAM. That's the budget I'm working around for the default Qwen3-8B
target and Qwen3-0.6B draft.

For another card, work through the same three pieces: weights, caches, and room
for the forward passes. Both models live entirely on one GPU in this version.
System RAM helps with loading and other work, but the loader doesn't use it to
extend VRAM or offload weights.

## start with the weights

BF16 stores each weight in two bytes. Using roughly 8.8 billion parameters for
the pair gives this weight estimate:

```text
target + draft ≈ 8.8 billion parameters
weights alone ≈ 17.6 billion bytes
              ≈ 16.4 GiB
```

That is only the weights. Leave room for both attention caches, activations,
attention workspace, logits, and the allocator. A card with 16 GiB of VRAM can't
hold this default BF16 pair even before those allocations.

The defaults are one request, a 4096-token total context budget, and up to 128
output tokens. The context budget includes the prompt, chat-template tokens, and
requested output. Model support for a longer context doesn't supply the memory
needed to use it.

## add the caches

The KV cache holds keys and values for tokens already processed. For one
full-attention model, a rough allocation is:

```text
2 × layers × tokens × KV heads × head dimension × bytes per value
```

The first `2` counts keys and values. Batch size would multiply this too; this
loop handles one request at a time.

The [target config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)
has 36 layers and 8 KV heads. The
[draft config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
has 28 layers and 8 KV heads. Both use head dimension 128. At BF16, the pair's
combined KV data works out to about 256 KiB per cached token, or 1 GiB at 4096
tokens. Temporary allocations and allocator overhead sit outside that estimate.

During verification, the target also processes the short proposed branch. The
[cache walkthrough](03-keeping-the-caches-honest.md) shows how that branch gets
cropped back to the committed prefix. Cropping changes logical lengths; a tensor
slice can retain its backing allocation until later operations replace it.

## if memory runs short

First, check `nvidia-smi` for other processes using the card. A desktop compositor
and other GPU applications share the same memory budget.

If the models load but generation runs out of memory, try a shorter prompt and
smaller output request. Lowering `--max-context` sets a tighter limit on those
requests; it doesn't preallocate a smaller cache by itself. Actual processed
sequence lengths determine cache growth. Fewer `--draft-tokens` can also reduce
the temporary verification block.

For example, with a short prompt:

```bash
specdecode --mode speculative \
  --prompt "Explain binary search in a few sentences." \
  --max-context 2048 \
  --max-new-tokens 64 \
  --draft-tokens 2
```

If the weights themselves don't fit, choose a smaller target. The
[Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) target with the existing
Qwen3-0.6B draft is a smaller pair to attempt:

```bash
specdecode --mode speculative \
  --target Qwen/Qwen3-1.7B \
  --draft Qwen/Qwen3-0.6B \
  --prompt "Explain binary search in a few sentences." \
  --max-context 2048 \
  --max-new-tokens 64
```

Changing model size changes the experiment. Use the same target in baseline mode
when comparing the loops. The loader requires dense, full-attention Qwen3 models
and checks vocabulary compatibility before loading both sets of weights.

`--dtype float16` is available for CUDA devices without BF16 support. It uses the
same two bytes per weight, so it doesn't halve the weight budget. Quantization
and CPU offload would need separate changes to the loader.

## keep the logits small

One memory choice in the code: for prefill and single-token steps, keep only the
last position's logits. Verification needs every position in its short input
block.

```python
logits_to_keep = 0 if all_positions else 1
```

In the pinned
[Qwen3 forward API](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py),
`0` means all positions. Keeping one position during prefill avoids building a
full prompt-length logits tensor when the loop only needs the next prediction.

## how this connects to setup

The [README](../README.md#set-up-the-environment) keeps the installation commands
in one place. I'm using a CUDA 12.8 PyTorch wheel and a fresh Python 3.12 virtual
environment. Transformers is pinned because the loop calls its cache API directly.

On Fedora, the versioned Python package lets the environment use a supported
interpreter alongside the system Python. KDE Plasma and Wayland don't change the
decoding loop; the relevant desktop detail here is their share of GPU memory.

The CLI checks CUDA before fetching model configs or weights. `specdecode --help`
only loads the argument parser, so you can inspect the controls without loading
the model stack.
