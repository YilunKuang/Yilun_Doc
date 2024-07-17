---
layout: default
title: Einsum
parent: Numerical Methods
---
# Einsum

Examples of einsum operations: 


```python

import torch

x = torch.rand((2,3))
print(f"x={x}")

# permutation of tensors
x_prime = torch.einsum("ij->ji",x)
print(f"x_prime={x_prime}")

# summation 
x_prime 

# column sum

# row sum

# matrix-vector multiplication

# matrix-matrix multiplication

# dot product first row with first row of matrix

# dot product with matrix 

# hadamard product (element-wise multiplication)

# outer product

# batch matrix multiplication

# matrix diagonal

# matrix trace
```


**Reference:**
- [1] [Einsum Is All You Need: NumPy, PyTorch and TensorFlow](https://www.youtube.com/watch?v=pkVwUVEHmfI)
