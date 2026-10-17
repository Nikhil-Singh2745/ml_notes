# Shapes first

I'll keep the batch dimension explicit. For this sketch:

| Symbol | Meaning | Toy size |
| --- | --- | --- |
| `B` | independent sequences | 2 |
| `T` | positions per sequence | 3 |
| `C` | input feature width | 4 |
| `D` | query/key width | 2 |
| `Dv` | value width | 2 |

These sizes are arbitrary. `x` contains feature vectors, not integer token IDs.

Three separate projections of the same input give queries, keys, and values:

```text
x:             (B, T, C)
Wq, Wk:        (C, D)
Wv:            (C, Dv)

q = x @ Wq:    (B, T, D)
k = x @ Wk:    (B, T, D)
v = x @ Wv:    (B, T, Dv)
```

The projection matrices here use mathematical right-multiplication notation. PyTorch's `nn.Linear` stores its weight in the transposed orientation, `(out_features, in_features)`, and applies it across the last input dimension. [Linear documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html).

The attention calculation is:

$$
S = \frac{QK^\top}{\sqrt{D}}, \qquad
A = \operatorname{softmax}(S + M), \qquad
Y = AV.
$$

`M` is the additive mask; [the next note](02-causal-mask.md) spells it out. This is scaled dot-product attention with a mask, following [section 3.2 of Attention Is All You Need](https://arxiv.org/html/1706.03762v7#S3.SS2).

The dimensions can be derived directly:

```text
q @ k.transpose(-2, -1): (B, T, D) @ (B, D, T)  -> (B, T, T)
weights @ v:            (B, T, T) @ (B, T, Dv) -> (B, T, Dv)
```

Row `i` of the score matrix belongs to query position `i`. Column `j` belongs to key position `j`. That orientation is worth fixing before drawing a mask.

For this three-dimensional case, matrix multiplication happens separately within each batch item. The last two dimensions are matrix dimensions; leading dimensions are batch dimensions. [PyTorch matmul rules](https://docs.pytorch.org/docs/2.14/generated/torch.matmul.html).

The output is another feature vector at each position. There isn't a vocabulary axis here, so these attention weights aren't next-token probabilities.

One useful distinction: queries and keys determine the weights. Values supply the vectors being mixed.
