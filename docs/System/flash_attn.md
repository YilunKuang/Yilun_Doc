---
layout: default

title: Flash Attention

parent: System
---

# Flash Attention

The main idea behind flash attention is reducing the numbers of High-Bandwidth Memory (HBM) access in the attention computation and thus achieving wall-clock time speedup at the expense of more FLOPs. In a high level, flash attention performs 1) **tiling** by computing fused attention operations (i.e. fusing all of the query-key product, softmax, dropout, masking, matmul with values in one kernel) by blocks with one HBM read and write, and 2) **selective gradient checkpointing** via recomputation of the intermediate query-key product and softmax during the backward pass to reduce the numbers of HBM read/write in the forward pass. 

## Online Softmax

For an input vector $$\mathbf{x}\in\mathbb{R}^d$$ with scaler entry $$\mathbf{x}_i\in\mathbb{R}$$, the standard softmax transformation is given by

$$
\begin{align}
\mathbf{y}&=\text{softmax}(\mathbf{x})\\
&=\frac{[\exp(\mathbf{x}_1),\cdots,\exp(\mathbf{x}_d)]}{\sum_{j=1}^{d}\exp(\mathbf{x}_j)}
\end{align}
$$

To prevent numerical overflow, we define the function $$m(\mathbf{x})=\max_j \mathbf{x}_j$$. Let $$f(\mathbf{x})=[\exp(\mathbf{x}_1-m(\mathbf{x})),\cdots,\exp(\mathbf{x}_d-m(\mathbf{x}))]$$ and $$\ell(\mathbf{x})=\sum_{j=1}^{d}\exp(\mathbf{x}_j-m(\mathbf{x}))=\sum_{j=1}^{d}f(\mathbf{x})_j$$. We can rewrite the computation as

$$
\begin{align}
\mathbf{y}=\frac{f(\mathbf{x})}{\ell(\mathbf{x})}
\end{align}
$$

This is called **safe softmax** since the maximum value of each entry of $$f(\mathbf{x})$$ now is just $$\exp(0)=1$$. 

It turns out that we can compute safe softmax in chunks. For the input vector $$\mathbf{x}\in\mathbb{R}^d$$, we can decompose it as a concatenation of two vectors $$\mathbf{x}=[\mathbf{x}^{(1)}\space\mathbf{x}^{(2)}]$$ where $$\mathbf{x}^{(1)},\mathbf{x}^{(2)}\in\mathbb{R}^{d/2}$$ and get

$$
\begin{align}
m(\mathbf{x})&=m([\mathbf{x}^{(1)}\space\mathbf{x}^{(2)}])=\max(m(\mathbf{x}^{(1)}),m(\mathbf{x}^{(2)}))\\
\ell(\mathbf{x})&=l([\mathbf{x}^{(1)}\space\mathbf{x}^{(2)}])=\exp(m(\mathbf{x}^{(1)})-m(\mathbf{x}))\ell(\mathbf{x}^{(1)})+\exp(m(\mathbf{x}^{(2)})-m(\mathbf{x}))\ell(\mathbf{x}^{(2)})\\
f(\mathbf{x})&=[\exp(m(\mathbf{x}^{(1)})-m(\mathbf{x}))f(\mathbf{x}^{(1)}),\cdots,\exp(m(\mathbf{x}^{(2)})-m(\mathbf{x}))f(\mathbf{x}^{(2)})]\\
\mathbf{y}&=\frac{f(\mathbf{x})}{\ell(\mathbf{x})}
\end{align}
$$

Thus it's possible to compute softmax one chunk at a time if we just reweight the chunked softmax output with stored global statistics $$m(\mathbf{x})$$ and $$\ell(\mathbf{x})$$. By doing chunked softmax computation sequentially, we obtain the **online softmax** algorithm. **Flash attention** basically uses the idea in online softmax for tiling as shown in Algorithm 1 in the flash attention paper.

**Reference**
- [1] Flash Attention 1. https://arxiv.org/pdf/2205.14135
- [2] Online normalizer calculation for softmax. https://arxiv.org/abs/1805.02867
- [3] Transformers Inference Optimization Toolset. https://astralord.github.io/posts/transformer-inference-optimization-toolset/