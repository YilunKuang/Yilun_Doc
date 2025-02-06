---
layout: default
title: Normalization
parent: Machine Learning
---

# Normalization

## LayerNorm

For an input vector $$\mathbf{x}\in\mathbb{R}^{d}$$, LayerNorm performs 1) re-centering by subtracting the mean $$\mathbb{E}[\mathbf{x}]$$ and 2) re-scaling by dividing the standard deviation $$\sqrt{\text{Var}[\mathbf{x}]}$$ as

$$
\mathbf{y}_{i}=\frac{\mathbf{x}_{i}-\mathbb{E}[\mathbf{x}]}{\sqrt{\text{Var}[\mathbf{x}]+\epsilon}}*\gamma_{i} + \beta_{i}
$$

where $$\epsilon$$ is a constant scaler for numerical stability and $$\gamma_i$$ and $$\beta_i$$ are element-wise learnable rescaling and shifting scalers.

## RMSNorm

RMSNorm performs re-scaling given by

$$
\mathbf{y}_{i} = \frac{\mathbf{x}_{i}}{\text{RMS}(\mathbf{x})}*\gamma_{i}
$$

where $$\text{RMS}(\mathbf{x})=\sqrt{\epsilon+\frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i^2}$$. [2] hypothesizes that re-centering is not needed and show empirically that RMSNorm achieves similar or better performance with less wall time compared to LayerNorm. 

## Pre-Normalization vs. Post-Normalization

See [3]


**Reference**
- [1] Layer Normalization. https://arxiv.org/abs/1607.06450 & https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- [2] Root Mean Square Layer Normalization. https://arxiv.org/abs/1910.07467 & https://pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html
- [3] On Layer Normalization in the Transformer Architecture. https://arxiv.org/abs/2002.04745


