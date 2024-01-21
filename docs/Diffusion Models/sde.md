---
layout: default

title: SDE

parent: Diffusion Models
---
# SDE

The diffusion process is governed by the solution to an Itô SDE:

$$
d\mathbf{x}=\mathbf{f}(\mathbf{x},t)+g(t)d\mathbf{w}
$$

with a corresponding reverse process

$$
d\mathbf{x}=[\mathbf{f}(\mathbf{x},t)-g(t)^2\nabla_{\mathbf{x}}\log p_t(\mathbf{x})]dt+g(t)d\mathbf{\bar{w}}
$$

