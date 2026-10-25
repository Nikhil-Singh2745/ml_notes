# choices in this version

I'm keeping this implementation small enough that I can follow every branch.
These are the decisions behind the defaults and the boundaries to keep in mind
when changing the code.

## greedy first

Both models choose `argmax`. No temperature, top-p, or repetition penalties.
Thinking mode is disabled in the chat template.

Greedy decoding makes the draft decision explicit: did the target choose the same
ID at this position? For general text generation, Qwen's
[model card](https://huggingface.co/Qwen/Qwen3-8B#best-practices) recommends sampling
settings and warns against greedy decoding in thinking mode. I'm keeping thinking
off here to focus on the generation machinery.

The full sampled algorithm needs probability ratios and a corrected residual
distribution after rejection. Replacing `argmax` with random sampling in this
loop wouldn't implement that algorithm.

## the reference behavior

In exact arithmetic, the target gets to choose every committed token. Accepted
draft tokens agree with it; the first disagreement gets its correction.
This gives the same output as the ordinary target-only greedy loop in that
exact-arithmetic setting.

With floating-point arithmetic, a block call and several single-token calls can
use different floating-point operation orders. If two scores are very close,
their ordering can flip. Greedy decoding doesn't make that impossible.

Also keeping `.eval()` and `torch.inference_mode()` in place. The first turns off
training behavior such as dropout. The second avoids building an autograd graph.
They have different jobs.

## model pair

Target: Qwen3-8B. Draft: Qwen3-0.6B.

I want both models to attach the same meaning to every ID. The loader compares
the complete token-to-ID mappings, special-token IDs, and logit vocabulary sizes.
Belonging to the same family or having equal vocabulary sizes alone wouldn't
establish that compatibility.

Only the target tokenizer builds the prompt. Both models consume those IDs.
The target's EOS IDs define stopping for the answer and draft rounds.

The loader accepts dense `qwen3` with full attention. Another architecture would
need its own checks for position handling, cache layout, and rollback behavior.

## keeping the code visible

I'm calling the models directly so the draft loop, verification call, prefix
comparison, and cache reconciliation stay together in
[speculative.py](../src/specdecode_lab/speculative.py).

Using PyTorch SDPA by default. `eager` is also selectable. No extra FlashAttention
package, compilation, CUDA graphs, or custom kernels in this version.

There are costs here: Python loops, token transfers to Python integers, dynamic
cache growth, and the final draft catch-up call after full acceptance.
The token trace adds decoding and printing too. I'm keeping those costs visible
while working through the decisions.

Questions left in my notes: how much bookkeeping could stay on the GPU? What
state would a probability-correct sampled version need to preserve?

