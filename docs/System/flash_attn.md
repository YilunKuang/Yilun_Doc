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

Thus it's possible to compute softmax one chunk at a time if we just reweight the chunked softmax output with stored global statistics $$m(\mathbf{x})$$ and $$\ell(\mathbf{x})$$. The following paragraph (from Flash Attn 2 paper) describes the chunked softmax algorithm in blocks: 

> For simplicity, consider just one row block of the attention matrix $$\mathbf{S}$$, of the form $$\begin{bmatrix} \mathbf{S}^{(1)} & \mathbf{S}^{(2)} \end{bmatrix}$$ for some matrices $$\mathbf{S}^{(1)}, \mathbf{S}^{(2)} \in \mathbb{R}^{B_r \times B_c}$$, where $$B_r$$ and $$B_c$$ are the row and column block sizes. We want to compute softmax of this row block and multiply with the value, of the form $$\begin{bmatrix} \mathbf{V}^{(1)} \\ \mathbf{V}^{(2)} \end{bmatrix}$$ for some matrices $$\mathbf{V}^{(1)}, \mathbf{V}^{(2)} \in \mathbb{R}^{B_c \times d}$$. Standard softmax would compute:


> $$
\begin{align}
m &= \max(\mathrm{rowmax}(\mathbf{S}^{(1)}), \mathrm{rowmax}(\mathbf{S}^{(2)})) \in \mathbb{R}^{B_r}  \\
\ell &= \mathrm{rowsum}(e^{\mathbf{S}^{(1)} - m}) + \mathrm{rowsum}(e^{\mathbf{S}^{(2)} - m}) \in \mathbb{R}^{B_r}  \\
\mathbf{P} &= \begin{bmatrix} \mathbf{P}^{(1)} & \mathbf{P}^{(2)} \end{bmatrix} = \text{diag}(\ell)^{-1}\begin{bmatrix} e^{\mathbf{S}^{(1)} - m} & e^{\mathbf{S}^{(2)} - m} \end{bmatrix} \in \mathbb{R}^{B_r \times 2B_c} \\
\mathbf{O} &= \begin{bmatrix} \mathbf{P}^{(1)} & \mathbf{P}^{(2)} \end{bmatrix} \begin{bmatrix} \mathbf{V}^{(1)} \\ \mathbf{V}^{(2)} \end{bmatrix} = \text{diag}(\ell)^{-1} e^{\mathbf{S}^{(1)} - m} \mathbf{V}^{(1)} + e^{\mathbf{S}^{(2)} - m} \mathbf{V}^{(2)} \in \mathbb{R}^{B_r \times d}.
\end{align} $$

By doing chunked softmax computation sequentially, we obtain the **online softmax** algorithm. For $$m_0=-\infty$$ and $$\ell_0=0$$, we have

$$
\begin{align}
m_i &= \max(m_{i-1},\mathbf{x}_i)\\
\ell_i &= \ell_{i-1}\exp(m_{i-1}-m_i)+\exp(\mathbf{x}_i-m_i)
\end{align}
$$

The more complicated version is given in the flash attention 2 paper:

> $$
\begin{align*}
m^{(1)} &= \mathrm{rowmax}(\mathbf{S}^{(1)})  \in \mathbb{R}^{B_r}\\
\ell^{(1)} &= \mathrm{rowsum}(e^{\mathbf{S}^{(1)} - m^{(1)}}) \in \mathbb{R}^{B_r} \\
\tilde{\mathbf{P}}^{(1)} &= \text{diag}(\ell^{(1)})^{-1} e^{\mathbf{S}^{(1)} - m^{(1)}} \in \mathbb{R}^{B_r \times B_c}\\
\mathbf{O}^{(1)} &= \tilde{\mathbf{P}}^{(1)} \mathbf{V}^{(1)} = \text{diag}(\ell^{(1)})^{-1} e^{\mathbf{S}^{(1)} - m^{(1)}} \mathbf{V}^{(1)} \in \mathbb{R}^{B_r \times d}\\
m^{(2)} &= \max(m^{(1)}, \mathrm{rowmax}(\mathbf{S}^{(2)})) = m \\
\ell^{(2)} &= e^{m^{(1)} - m^{(2)}} \ell^{(1)} + \mathrm{rowsum}(e^{\mathbf{S}^{(2)} - m^{(2)}}) = \mathrm{rowsum}(e^{\mathbf{S}^{(1)} - m}) + \mathrm{rowsum}(e^{\mathbf{S}^{(2)} - m}) = \ell \\
\tilde{\mathbf{P}}^{(2)} &= \text{diag}(\ell^{(2)})^{-1} e^{\mathbf{S}^{(2)} - m^{(2)}} \\
\mathbf{O}^{(2)} &= \text{diag}(\ell^{(1)} / \ell^{(2)})^{-1} \mathbf{O}^{(1)} + \tilde{\mathbf{P}}^{(2)} \mathbf{V}^{(2)} = \text{diag}(\ell^{(2)})^{-1} e^{s^{(1)} - m} \mathbf{V}^{(1)} + \text{diag}(\ell^{(2)})^{-1} e^{s^{(2)} - m} \mathbf{V}^{(2)} = \mathbf{O}.
\end{align*}
$$

The online softmax algorithm enables tiling as shown in Algorithm 1 in the **flash attention**  paper.

## Flash Attention 1

TODO

## Flash Attention 2

### Algorithm 

The scaling by the $$\ell$$ are delayed. $$m$$ and $$\ell$$ are simplified with the logsumexp trick. Causal masking is also replaced by skipping computations. 

### Parallelism

In flash attention 1, each thread block executes one attention head. So in total there are batch size * number of heads many thread blocks. Each thread block is scheduled on one streaming multiprocessor (SM). For A100 with 108 SMs, the GPU is most efficiently utilized for larger batch sizes or number of heads. 

However, for long context inference with small batch size, it makes more sense to do sequence parallelism to fully utilize the 108 SMs. So in flash attention 2, the loop over sequence length is now the outer loop as opposed to being the inner loop in flash attention 1. The loop over the sequence length can thus be scheduled over different thread blocks. Thus in total there are batch size * number of heads * number of subsequences many thread blocks or something like that during the forward pass. For the backward pass, flash attention 2 ensures that each thread block compute one block of the columns of the attention matrix. 

### Work Partitioning Between Warps

In flash attention 1, for each thread block, the $$\mathbf{K}$$ and $$\mathbf{V}$$ are split and the $$\mathbf{Q}$$ are accessible by all warps inside a thread block. In flash attention 2, $$\mathbf{Q}$$ are split into warps and 

**Reference**
- [1] Flash Attention 1. https://arxiv.org/pdf/2205.14135
- [2] Flash Attention 2. https://arxiv.org/abs/2307.08691
- [3] Online normalizer calculation for softmax. https://arxiv.org/abs/1805.02867
- [4] Transformers Inference Optimization Toolset. https://astralord.github.io/posts/transformer-inference-optimization-toolset/
