# Three positions, worked by hand

Constructed values, not output from a run. I'll use one batch item and query/key width `D = 1`, so the scale factor is just `1`.

Choose:

```text
q = [[0],       k = [[0],       v = [[2, 0],
     [1],            [ln(2)],        [0, 2],
     [1]]            [ln(4)]]       [4, 4]]
```

These are chosen projected vectors, not learned parameters or token embeddings. Values have width `Dv = 2`.

Before masking:

$$
S = QK^\top =
\begin{bmatrix}
0 & 0 & 0 \\
0 & \ln 2 & \ln 4 \\
0 & \ln 2 & \ln 4
\end{bmatrix}.
$$

After applying the causal mask:

$$
S + M =
\begin{bmatrix}
0 & -\infty & -\infty \\
0 & \ln 2 & -\infty \\
0 & \ln 2 & \ln 4
\end{bmatrix}.
$$

Exponentiating row by row gives `[1, 0, 0]`, `[1, 2, 0]`, and `[1, 2, 4]`. Divide each row by its sum:

$$
A =
\begin{bmatrix}
1 & 0 & 0 \\
1/3 & 2/3 & 0 \\
1/7 & 2/7 & 4/7
\end{bmatrix}.
$$

This uses the ordinary [softmax definition](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html); the fractions follow directly from the constructed scores.

Now mix the value vectors:

```text
y[0] = 1 * [2, 0]
     = [2, 0]

y[1] = (1/3) * [2, 0] + (2/3) * [0, 2]
     = [2/3, 4/3]

y[2] = (1/7) * [2, 0] + (2/7) * [0, 2] + (4/7) * [4, 4]
     = [18/7, 20/7]
```

The first position returns its own value vector. There's only one allowed key, so its weight must be `1` for any finite score.

The second and third queries are identical. Their outputs differ because the third query has another visible key/value pair. The visible set affects both what can contribute and the normalization denominator.

Possible follow-up on paper: change only the last value vector. Which output rows can change? Only the last row here; the earlier rows assign it zero weight.
