# keeping the caches honest

At each round boundary:

```text
committed prefix: [prompt ... generated ... pending]
target cache:     [prompt ... generated ...        ]
draft cache:      [prompt ... generated ...        ]
```

Both caches contain the committed prefix except its last token.
`pending` holds that last token separately.

Code: [CachedModel](../src/specdecode_lab/decoding.py) and
[cache reconciliation](../src/specdecode_lab/speculative.py).

## starting the loop

Target prefill caches the prompt and predicts the first output token.
Commit that token as the seed. It's now pending.

If generation continues, prefill the draft with the same prompt. Ignore the
draft's prediction here. The target already picked the seed.

Now both caches stop just before pending, which is the boundary every later round
returns to.

## one round, with lengths

Say the committed prefix is 101 tokens long and both caches hold 100. Pending is
the token at position 101. Ask for four proposals.

| Point                        | Target cache length | Draft cache length |
| ---------------------------- | ------------------- | ------------------ |
| Before drafting              | 100                 | 100                |
| Four proposals produced      | 100                 | 104                |
| Target verification finished | 105                 | 104                |

The draft processed `[pending, d1, d2, d3]` to predict `[d1, d2, d3, d4]`.
It hasn't processed `d4` yet.

The target processed `[pending, d1, d2, d3, d4]` in its verification call.
That explains the one-token difference.

### if only two proposals match

Commit `[d1, d2, correction]`. Prefix length becomes 104.
The new pending token is the correction. Both caches should hold 103 tokens:
the old 100, the old pending, `d1`, and `d2`.

```python
keep_length = prompt_ids.shape[1] + len(generated) - 1
target_state.crop(keep_length)
draft_state.crop(keep_length)
```

Cropping is essential. A rejected token left in the cache would keep influencing
later output even after I removed it from the displayed answer.

### if all four match and there's room for a bonus

Commit four proposals plus the bonus. Prefix length becomes 106.
The bonus is pending. Both caches need 105 tokens.

The target is already there, while the draft is still at 104. Feed `d4` through
the draft once to catch it up. Ignore the resulting prediction; the target chose
the bonus.

That extra draft call belongs in the cost of a round too. Reconciliation also
runs on the final round, even when the caches won't be reused.

## positions after rollback

The next position is derived from the cropped cache length:

```python
positions = torch.arange(cache_length, cache_length + input_length, device=device)
```

A separate ever-increasing position counter would be wrong after rejection.
The next token must continue from the accepted branch.

I'm using Transformers' `DynamicCache.crop()` for this. The pinned
[cache implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/cache_utils.py)
has the full-attention crop behavior this loop depends on.
Sliding-window and hybrid caches need separate reasoning.

Cropping changes logical lengths. A tensor slice can still retain its old
allocation until later operations replace it, so a shorter cache doesn't imply
an immediate VRAM refund.

In the trace, `both_caches` shows the lengths before the round and after
reconciliation. To connect the numbers to the code, find `base_length` and
`keep_length` in [speculative.py](../src/specdecode_lab/speculative.py). The
boundary check at the start of each round enforces this invariant.
