# checking a draft

Code: [generate_speculative](../src/specdecode_lab/speculative.py).

The small model proposes up to `draft_tokens` tokens. It still generates them
one at a time. The bigger model checks that short branch in one forward pass.

The useful bit is that several target positions get evaluated together. Drafting
still costs a sequence of small-model calls, so the payoff depends on how many
proposals the target accepts and the relative cost of those calls.

## lining up the positions

`pending` is the last token already committed to the answer. It hasn't gone
through either cache yet. More on that in [the cache note](03-keeping-the-caches-honest.md).

For three proposals:

```text
target input:       pending   d1        d2        d3
prediction from:   pending   d1        d2        d3
predicts:          d1        d2        d3        bonus
```

Each input position predicts the token after it. This is where I need to watch
for an index shifted by one: the first returned prediction gets compared with `d1`.

```python
verification_input = torch.cat([pending, *proposal_tensors], dim=1)
target_predictions = target_state.predict(
    verification_input, all_positions=True
)[0].tolist()
```

The causal attention mask still applies. The score used to judge `d1` can't
peek at `d1` or later proposals. Parallel evaluation doesn't remove causality.

## stop at the first mismatch

Made-up IDs, just to follow the decision:

```text
draft:              10   20   30   40
target predictions: 10   20   99   40   50
commit:             10   20   99
```

The last `40` matching means nothing now. Its target prediction was conditioned
on a rejected `30`. After committing `99`, that branch is gone.

So I accept only a contiguous matching prefix, then append the target's
prediction at the mismatch. Scanning ahead for more matches would keep tokens
from a branch the target has already rejected.

If everything matches, the last verification position already predicts one
more token. The loop appends that as a bonus if the output budget allows it.

## stopping

A draft EOS ends drafting for that round. It doesn't end the answer unless the
target accepts it. A correction or bonus can also be EOS.

The proposal count shrinks near the output cap, and the loop skips the bonus when
it would exceed that cap. Every round commits at least one token, including a
round where the first proposal is rejected.

To inspect these decisions, run the [trace command](../README.md#follow-a-round-in-the-trace)
and follow one round's `accept`, `reject`, and `correct` labels. A round with full
acceptance can show `bonus` instead. The next piece is making sure both caches
follow the [committed branch](03-keeping-the-caches-honest.md).

The [speculative decoding paper](https://arxiv.org/abs/2211.17192) is the source
of the draft/parallel-verification idea. This implementation sticks to greedy
token agreement. The probability-based sampler is a different piece of work.

