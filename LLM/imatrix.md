Suppose a tensor row is represented as:

$$ \mathbf{w} = [w_1, w_2, \ldots, w_d] $$

and an activation column is represented as:

$$ \mathbf{a} = [a_1, a_2, \ldots, a_d].  $$

The dot product of $\mathbf{w}$ and $\mathbf{a}$ produces one entry in the activation passed to the next layer:

$$ y = \mathbf{w} \cdot \mathbf{a} = \sum_{j=1}^d w_j a_j.  $$

Suppose the quantized version of $\mathbf{w}$ is:

$$ \mathbf{q} = [q_1, q_2, \ldots, q_d].  $$

The quantized values should minimize the error in the dot product due to quantization. The error function can be written as:

$$ F = \left[ \sum_{j=1}^d (q_j - w_j) a_j \right]^2.  $$

Define $ r_j = q_j - w_j $, then:

$$ F = \left[\sum_j a_j r_j\right]^2.  $$

Since there are multiple activations, we can consider the expectation of the error:

$$ \mathbb{E}[F] = \mathbb{E}\left[\left(\sum_j a_j r_j\right)^2\right].  $$

Expanding this expression:

$$ \mathbb{E}[F] = \sum_j \mathbb{E}[a_j^2] r_j^2 + \sum_{i \neq j} \mathbb{E}[a_i a_j] r_i r_j.  $$

If the activations are not strongly correlated (i.e., $\mathbb{E}[a_i a_j] \approx 0$ for $i \neq j$), the second term can be neglected, yielding:

$$ \mathbb{E}[F] \approx \sum_j \mathbb{E}[a_j^2] r_j^2 = \sum_j \langle a_j^2 \rangle (q_j - w_j)^2.  $$

Minimizing the weighted mean square error $\sum_j \langle a_j^2 \rangle (q_j - w_j)^2$ is equivalent to minimizing $\mathbb{E}[F]$. Therefore, the expectation $\mathbb{E}[\mathbf{a^2}]$ can be used as an importance matrix to quantize $\mathbf{w}$.
