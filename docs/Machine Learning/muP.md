---
layout: default
title: muP
parent: Machine Learning
---
# muP

## Lindeberg–Lévy Central Limit Theorem (CLT)

If $$X_1, X_2, X_3, ...$$ is a sequence of i.i.d random variables with $$\mathbb{E}[X_i]=\mu$$ and $$\text{Var}[X_i]=\sigma^2<\infty$$, then as $$n\to\infty$$ the random variables $$\sqrt{n}(\bar{X}_n-\mu)$$ converge in distribution to a normal $$\mathcal{N}(0,\sigma^2)$$:

$$
\sqrt{n}(\bar{X}_n-\mu)\xrightarrow{d}\mathcal{N}(0,\sigma^2)
$$

where $$\bar{X}_n:=\frac{X_1+...+X_n}{n}$$. If $$\mu=0$$, we can also write

$$
\frac{1}{\sqrt{n}}\sum_{i=1}^{n}X_i\xrightarrow{d}\mathcal{N}(0,\sigma^2)

$$

## Weak Law of Large Numbers (LLN)
If $$X_1, X_2, X_3, ...$$ is a sequence of i.i.d random variables with $$\mathbb{E}[X_i]=\mu<\infty$$, then as $$n\to\infty$$ the sample mean $$\bar{X}_n$$ converges in probability to $$\mu$$:

$$
\begin{align}
\bar{X}_n &\xrightarrow{p}\mu \\
\iff \frac{1}{n}\sum_{i=1}^{n}X_i &\xrightarrow{p}\mathbb{E}[X_i]
\end{align}
$$

## Sizes of Summation under Scaling

Given CLT and LLN, the basic intuition of the scale of a summation over $$X_i$$ is given by

$$
\text{when } n \text{ is large, }\sum_{i=1}^{n}X_n \text{ has typical size}
\begin{cases}
  \Theta(n) & \text{if } \mathbb{E}[X]\neq0 \text{ (LLN)}\\    
  \Theta(\sqrt{n}) & \text{if } \mathbb{E}[X]=0  \text{ (CLT)}\\
\end{cases}
$$

## Gaussian Matrices and Tensor Products

Here is the expected entry size of $$Av$$ for different matrices $$A$$ and vector $$v$$ correlated with each other, both having entries of size $$\Theta(1)$$:

|       | Standard Gaussian <br> $$A\in\mathbb{R}^{n\times n}$$ | (Nonlinear) Tensor Product <br> $$A\in\mathbb{R}^{n\times n}$$ | Vector <br> $$A\in\mathbb{R}^{1\times n}$$ |
|:-----:|:-----------------------------------------------------:|:--------------------------------------------------------------:|:-------------------------------------------:|
|Entry size <br> of $$Av$$ | $$\Theta(\sqrt{n})$$ | $$\Theta(n)$$ | $$\Theta(n)$$ |

## Desiderata of Parametrization
1. Every (pre)activation vector in a network should have $$\Theta(1)$$-sized coordinates;
2. Neural network output should be $$\mathcal{O}(1)$$;
3. All parameters should be updated as much as possible (in terms of scaling in width) without leading to divergence

**Reference**
- Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao. Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer. https://arxiv.org/abs/2203.03466.