---
layout: default

title: Flash Attention

parent: System
---

# Flash Attention

The main idea behind flash attention is reducing the numbers of High-Bandwidth Memory (HBM) access in the attention computation and thus achieving wall-clock time speedup at the expense of more FLOPs. In a high level, flash attention performs 1) **tiling** by computing fused attention operations (i.e. fusing all of the query-key product, softmax, dropout, masking, matmul with values in one kernel) by blocks with one HBM read and write, and 2) **selective gradient checkpointing** via recomputation of the intermediate query-key product and softmax during the backward pass to reduce the numbers of HBM read/write in the forward pass. 

## Safe Softmax

For an input vector $$\mathbf{x}\in\mathbb{R}^d$$ with scaler entry $$\mathbf{x}_i\in\mathbb{R}$$, the standard softmax transformation is given by

$$
\mathbf{y}_i=\frac{\exp(\mathbf{x}_i)}{\sum_{j=1}^{d}\exp(\mathbf{x}_j)}
$$

To prevent numerical overflow, we define the function $$m(\mathbf{x})=\max_j \mathbf{x}_j$$ and 