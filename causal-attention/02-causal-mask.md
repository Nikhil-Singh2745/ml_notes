# What the mask actually changes

Fix the convention first: rows are queries, columns are keys.

For a sequence of length four, position `i` may use positions `j <= i`. The allowed connections are:

```text
             key position
             0  1  2  3
query  0     1  0  0  0
       1     1  1  0  0
       2     1  1  1  0
       3     1  1  1  1
```

The corresponding additive mask has `0` for allowed entries and `-inf` for blocked ones. Add it to scores **before** softmax. This is also the square causal convention in [PyTorch's scaled dot-product attention reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

## Zeroing a score doesn't block it

Softmax exponentiates its inputs. A zero score contributes `exp(0) = 1` to the denominator. A blocked entry needs zero probability mass, represented mathematically by `exp(-inf) = 0`.

Softmax runs across the keys, `dim=-1`, so each query distributes its weight among its allowed positions. [Softmax definition and dimension argument](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html).

Multiplying probabilities by a mask afterward would remove entries without automatically redistributing their mass. That's a different calculation unless the remaining probabilities are renormalized.

## Why the diagonal stays

Consider this constructed next-token setup, using letters as token IDs:

```text
input:   A  B  C
target:  B  C  D
```

At input position `1`, `B` is already available. The prediction there is for `C`. Letting that position use its own input doesn't expose the answer; letting it read input position `2` would.

This assumes earlier layers also preserve causality. A mask can't undo future information that was already mixed into its inputs.

## Limits of this sketch

There are no padding tokens or packed independent documents here. Those would need additional visibility rules. Every row in this square mask has at least one allowed entry. Blocking an entire row leaves no probability distribution to normalize; don't casually extend the hand-written softmax to that case.

Also keep the meaning of a Boolean mask explicit. In `scaled_dot_product_attention`, `True` means allowed. In the later code sketch, `True` instead marks entries to overwrite with `-inf`.
