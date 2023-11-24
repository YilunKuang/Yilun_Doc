---
layout: default

title: Score Matching

parent: Diffusion Models
---
**Score Matching**

Consider the task of density estimation for a random vector $$\mathbf{x}\in\mathbb{R}^{n}$$ with a probability density function $$p_\mathbf{x}(\cdot)$$. Suppose we have a parametrized density model $$p(\boldsymbol{\xi};\boldsymbol{\theta})=\frac{1}{Z(\boldsymbol{\theta})}q(\boldsymbol{\xi};\boldsymbol{\theta})$$ with parameters $$\boldsymbol{\theta}\in\mathbb{R}^{m}$$ and an intractable normalization constant $$Z(\boldsymbol{\theta}):=\int_{\boldsymbol{\xi}\in\mathbb{R}^n}q(\boldsymbol{\xi};\boldsymbol{\theta})d\boldsymbol{\xi}$$. 

To avoid the numerical approximation of the integral $$\int_{\boldsymbol{\xi}\in\mathbb{R}^n}q(\boldsymbol{\xi};\boldsymbol{\theta})d\boldsymbol{\xi}$$ when $$n\gg1$$, we use the score function of the parametrized model defined as

$$
\boldsymbol{\psi}(\boldsymbol{\xi};\boldsymbol{\theta}):=\nabla_{\boldsymbol{\xi}}\log p(\boldsymbol{\xi};\boldsymbol{\theta})=\nabla_{\boldsymbol{\xi}}\log q(\boldsymbol{\xi};\boldsymbol{\theta})-\nabla_{\boldsymbol{\xi}}\log Z(\boldsymbol{\theta})=\nabla_{\boldsymbol{\xi}}\log q(\boldsymbol{\xi};\boldsymbol{\theta})
$$

and the score function of the data distribution $$\boldsymbol{\psi}_{\mathbf{x}}(\cdot):=\nabla_{\boldsymbol{\xi}}\log p_{\mathbf{x}}(\cdot)$$. We can estimate the density by minimizing the expected square distance between $$\boldsymbol{\psi}(\cdot;\boldsymbol{\theta})$$ and $$\boldsymbol{\psi}_{\mathbf{x}}(\cdot)$$:

$$
J(\boldsymbol{\theta})=\frac{1}{2}\int_{\boldsymbol{\xi}\in\mathbb{R}^n}p_\mathbf{x}(\boldsymbol{\xi})\|\boldsymbol{\psi}(\boldsymbol{\xi};\boldsymbol{\theta})-\boldsymbol{\psi}_{\mathbf{x}}(\boldsymbol{\xi})\|^2d\boldsymbol{\xi}

$$

$$
\hat{\boldsymbol{\theta}}=\text{argmin}_{\boldsymbol{\theta}}J(\boldsymbol{\theta})
$$

This is still non-trivial since we need to estimate $$\boldsymbol{\psi}_{\mathbf{x}}(\cdot)$$. It’s shown in [1] that under some weak regularity conditions $$J$$ is exactly equal to

$$
J(\boldsymbol{\theta})=\int_{\boldsymbol{\xi}\in\mathbb{R}^n}p_\mathbf{x}(\boldsymbol{\xi})\bigg[\frac{1}{2}\|\boldsymbol{\psi}(\boldsymbol{\xi};\boldsymbol{\theta})\|^2+\text{tr}(\nabla_{\boldsymbol{\xi}}\boldsymbol{\psi}(\boldsymbol{\xi};\boldsymbol{\theta}))\bigg]d\boldsymbol{\xi}+const
$$

This new objective only involves the computation of the score function of the parametrized model $$\boldsymbol{\psi}(\boldsymbol{\xi};\boldsymbol{\theta})$$. The proof idea is to expand the square norm difference and substitute the cross term with the trace term via partial integration. 

This result is presented in [2] as a basis of score estimation for diffusion models.

**Reference**

[1] Hyvärinen, A., & Dayan, P. (2005). Estimation of Non-normalized Statistical Models by Score Matching. Journal of Machine Learning Research, 6(4).

[2] Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. Advances in Neural Information Processing Systems, 32.