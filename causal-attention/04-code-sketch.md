# A small PyTorch sketch

Code sketches in this note are untested.

Keep one head, no padding, no dropout, and no cache. Assume nonempty inputs with finite scores, using ordinary floating-point tensors. Queries, keys, and values use the same sequence length and batch size.

```python
import torch
from torch import nn


def causal_attention(q, k, v):
    # q, k: (B, T, D); v: (B, T, Dv)
    scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    length = q.shape[-2]

    positions = torch.arange(length, device=q.device)
    # Row i, column j: block when j > i.
    future = positions[None, :] > positions[:, None]
    scores = scores.masked_fill(future, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class OneHeadAttention(nn.Module):
    def __init__(self, input_width, head_width, value_width):
        super().__init__()
        self.query = nn.Linear(input_width, head_width, bias=False)
        self.key = nn.Linear(input_width, head_width, bias=False)
        self.value = nn.Linear(input_width, value_width, bias=False)

    def forward(self, x):
        # x: (B, T, input_width)
        return causal_attention(self.query(x), self.key(x), self.value(x))
```

`future` has shape `(T, T)`. It broadcasts across the batch axis. Its name describes what `True` means: overwrite that score. This follows the mask-replacement operation documented for [`masked_fill`](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.masked_fill.html).

The three projections are separate learned layers. Each maps the last input dimension while preserving the leading dimensions. [Linear reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html).

Returning weights is useful for following the calculation. The [shape note](01-shapes.md) gives the expected dimensions; the [hand-worked example](03-by-hand.md) gives a small mathematical reference. Neither is evidence that this implementation has been executed.

## The library equivalent

For this same square, single-head setup, the attention calculation can instead be expressed as:

```python
import torch.nn.functional as F


def causal_attention_sdpa(q, k, v):
    # Add an explicit head axis: (B, 1, T, D).
    output = F.scaled_dot_product_attention(
        q.unsqueeze(1),
        k.unsqueeze(1),
        v.unsqueeze(1),
        is_causal=True,
        dropout_p=0.0,
    )
    return output.squeeze(1)
```

This returns the mixed values, without the attention matrix. `dropout_p=0.0` is explicit because this function applies dropout according to that argument, independently of a surrounding module's evaluation mode. The mathematical operation matches the manual sketch under these assumptions; floating-point results need not be bit-for-bit identical across implementations. [SDPA reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

This is only the attention part. There are no positional features, output projection, residual path, normalization, feed-forward layer, or prediction objective. The first module constructs the Q/K/V projections; it doesn't constitute a complete transformer block. [Transformer architecture, section 3](https://arxiv.org/html/1706.03762v7#S3).

Possible follow-up: split the projected width into several heads, then write down the shape changes before touching the code.
