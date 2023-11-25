---
layout: default

title: Score Matching

parent: Diffusion Models
---

# Score Matching

## Background: Density Estimation

Consider the task of density estimation for a random vector $$\mathbf{x}\in\mathbb{R}^{n}$$ with a probability density function $$p_\mathbf{x}(\cdot)$$. Suppose we have a parametrized density model $$p(\boldsymbol{\xi};\boldsymbol{\theta})=\frac{1}{Z(\boldsymbol{\theta})}q(\boldsymbol{\xi};\boldsymbol{\theta})$$ with parameters $$\boldsymbol{\theta}\in\mathbb{R}^{m}$$ and an intractable normalization constant $$Z(\boldsymbol{\theta}):=\int_{\boldsymbol{\xi}\in\mathbb{R}^n}q(\boldsymbol{\xi};\boldsymbol{\theta})d\boldsymbol{\xi}$$. 

## Score Matching

To avoid the numerical approximation of the integral $$\int_{\boldsymbol{\xi}\in\mathbb{R}^n}q(\boldsymbol{\xi};\boldsymbol{\theta})d\boldsymbol{\xi}$$ when $$n\gg1$$, we use the score function of the parametrized model defined as

$$
\boldsymbol{\psi}(\boldsymbol{\xi};\boldsymbol{\theta}):=\nabla_{\boldsymbol{\xi}}\log p(\boldsymbol{\xi};\boldsymbol{\theta})=\nabla_{\boldsymbol{\xi}}\log q(\boldsymbol{\xi};\boldsymbol{\theta})-\nabla_{\boldsymbol{\xi}}\log Z(\boldsymbol{\theta})=\nabla_{\boldsymbol{\xi}}\log q(\boldsymbol{\xi};\boldsymbol{\theta})
$$

and the score function of the data distribution $$\boldsymbol{\psi}_{\mathbf{x}}(\cdot):=\nabla_{\boldsymbol{\xi}}\log p_{\mathbf{x}}(\cdot)$$. We can estimate the density by minimizing the expected square distance between $$\boldsymbol{\psi}(\cdot;\boldsymbol{\theta})$$ and $$\boldsymbol{\psi}_{\mathbf{x}}(\cdot)$$ via the explicit score matching objective

$$
J_{\text{ESM}}(\boldsymbol{\theta})=\frac{1}{2}\int_{\boldsymbol{\xi}\in\mathbb{R}^n}p_\mathbf{x}(\boldsymbol{\xi})\|\boldsymbol{\psi}(\boldsymbol{\xi};\boldsymbol{\theta})-\boldsymbol{\psi}_{\mathbf{x}}(\boldsymbol{\xi})\|^2d\boldsymbol{\xi}=\mathbb{E}_{p_{\mathbf{x}}(\boldsymbol{\xi})}\bigg[\frac{1}{2}\|\boldsymbol{\psi}(\boldsymbol{\xi};\boldsymbol{\theta})-\boldsymbol{\psi}_{\mathbf{x}}(\boldsymbol{\xi})\|^2\bigg]
$$

$$
\hat{\boldsymbol{\theta}}=\text{argmin}_{\boldsymbol{\theta}}J_{\text{ESM}}(\boldsymbol{\theta})
$$

This is still non-trivial since we need to estimate $$\boldsymbol{\psi}_{\mathbf{x}}(\cdot)$$. It’s shown in [1] that under some conditions ($$p_\mathbf{x}(\cdot)$$ is differentiable and the log density is finite everywhere) $$J$$ is asymptotically equivalent to

$$
\tilde{J}(\boldsymbol{\theta})=\int_{\boldsymbol{\xi}\in\mathbb{R}^n}p_\mathbf{x}(\boldsymbol{\xi})\bigg[\frac{1}{2}\|\boldsymbol{\psi}(\boldsymbol{\xi};\boldsymbol{\theta})\|^2+\text{tr}(\nabla_{\boldsymbol{\xi}}\boldsymbol{\psi}(\boldsymbol{\xi};\boldsymbol{\theta}))\bigg]d\boldsymbol{\xi}+const
$$

$$
=\mathbb{E}_{p_{\mathbf{x}}(\boldsymbol{\xi})}\bigg[\frac{1}{2}\|\boldsymbol{\psi}(\boldsymbol{\xi};\boldsymbol{\theta})\|^2+\text{tr}(\nabla_{\boldsymbol{\xi}}\boldsymbol{\psi}(\boldsymbol{\xi};\boldsymbol{\theta}))\bigg]+const
$$

under the law of large numbers. This new objective only involves the computation of the score function of the parametrized model $$\boldsymbol{\psi}(\boldsymbol{\xi};\boldsymbol{\theta})$$. The proof idea is to expand the square norm difference and substitute the cross term with the trace term via partial integration. 

## Denoising Score Matching

In many real world scenarios, $$\tilde{J}(\boldsymbol{\theta})$$ gives a poor estimation since 1) we’re always in the non-asymptotic regime and 2) the distribution for finite images are discontinuous hence not differentiable [2]. Also, we would like to avoid computing $$\text{tr}(\nabla_{\boldsymbol{\xi}}\boldsymbol{\psi}(\boldsymbol{\xi};\boldsymbol{\theta}))$$ when our parametrized model $$q(\boldsymbol{\xi};\boldsymbol{\theta})$$ is a deep neural network. 

There is an alternative method called denoising score matching where the input data $$\boldsymbol{\xi}$$ is perturbed with $$\boldsymbol{\epsilon}\sim\mathcal{N}(0,\sigma^2\mathbf{I})$$ such that $$\tilde{\boldsymbol{\xi}}=\boldsymbol{\xi}+\boldsymbol{\epsilon},$$ $$\tilde{\boldsymbol{\xi}}\sim\mathcal{N}(\boldsymbol{\xi},\sigma^2\mathbf{I})$$ [3]. We can the perturbed data distribution $$p_\sigma(\tilde{\boldsymbol{\xi}}\vert\boldsymbol{\xi})$$. Thus the score function of the perturbed data is given by

$$
\boldsymbol{\psi}_{\tilde{\mathbf{x}}}(\tilde{\boldsymbol{\xi}})=\nabla_{\tilde{\boldsymbol{\xi}}}\log p_\sigma(\tilde{\boldsymbol{\xi}}\vert\boldsymbol{\xi})=\frac{1}{2}(\boldsymbol{\xi}-\tilde{\boldsymbol{\xi}})
$$

We can then replace $$\boldsymbol{\psi}_{\mathbf{x}}(\boldsymbol{\xi})$$ in $$J_{\text{ESM}}$$ with $$\boldsymbol{\psi}_{\tilde{\mathbf{x}}}$$ and get the equivalent minimization objective $$J_{\text{DSM}}$$:

$$
J_{\text{DSM}}(\boldsymbol{\theta})=\frac{1}{2}\int_{\tilde{\boldsymbol{\xi}}\in\mathbb{R}^n}\int_{\boldsymbol{\xi}\in\mathbb{R}^n}p_\sigma(\tilde{\boldsymbol{\xi}},\boldsymbol{\xi})\|\boldsymbol{\psi}(\tilde{\boldsymbol{\xi}};\boldsymbol{\theta})-\boldsymbol{\psi}_{\tilde{\mathbf{x}}}(\tilde{\boldsymbol{\xi}})\|^2d\boldsymbol{\xi}d\tilde{\boldsymbol{\xi}}
=\mathbb{E}_{p_\sigma(\tilde{\boldsymbol{\xi}},\boldsymbol{\xi})}\bigg[\frac{1}{2}\|\boldsymbol{\psi}(\tilde{\boldsymbol{\xi}};\boldsymbol{\theta})-\boldsymbol{\psi}_{\tilde{\mathbf{x}}}(\tilde{\boldsymbol{\xi}})\|^2\bigg]
$$

where $$p_\sigma(\tilde{\boldsymbol{\xi}},\boldsymbol{\xi})=p_\sigma(\tilde{\boldsymbol{\xi}}\vert\boldsymbol{\xi})p_{\mathbf{x}}(\boldsymbol{\xi})$$ is the joint density. The proof of equivalence involves expanding the squared norm differences and replacing the cross term. It’s also shown that given a properly chosen energy function $$E(\boldsymbol{\xi};\boldsymbol{\theta})$$ for the parametrized model $$p(\boldsymbol{\xi};\boldsymbol{\theta})=\frac{1}{Z(\boldsymbol{\theta})}\exp (-E(\boldsymbol{\xi};\boldsymbol{\theta}))$$, the denoising auto-encoder (DAE) training objective is equivalent to performing score matching [3]

This result is presented in [4] as a basis of denoising score estimation for diffusion models. 

**Reference**

[1] Hyvärinen, A., & Dayan, P. (2005). Estimation of Non-normalized Statistical Models by Score Matching. Journal of Machine Learning Research, 6(4).

[2] Kingma, D. P., and Cun, Y. (2010). "Regularized Estimation of Image Statistics by Score Matching." In Advances in Neural Information Processing Systems, Vol. 23.

[3] Vincent, P. (2011). "A Connection Between Score Matching and Denoising Autoencoders." *Neural Computation*, 23(7), 1661-1674.

[4] Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. Advances in Neural Information Processing Systems, 32.