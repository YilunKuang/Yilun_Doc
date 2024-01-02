---
layout: default
title: Density Estimation
parent: Machine Learning
---
# Density Estimation

## Question

The successes of both Large Language Models and Diffusion Models are based on the idea of density estimation. Why is density estimation so powerful? 

## Empirical Dataset Distribution

Consider a set of data points $$\{ \mathbf{x}_{i}\}_{i=1}^{N}$$ where $$\mathbf{x}_{i}\in\mathbb{R}^{d}$$. 

Without paying any extra efforts, an extremely naive density estimation approach is simply representing the empirical datapoints distribution as a mixture of Dirac delta distribution. Recall that a Dirac delta function $$\delta(\cdot)$$ is defined as

$$
\delta(\mathbf{x}) =
\begin{cases} 
+ \infty, & \mathbf{x} = 0 \\
0, & \mathbf{x} \neq 0 
\end{cases}
$$

under the constraint $$\int_{-\infty}^{+\infty}\delta(\mathbf{x})dx=1$$. Thus our empirical data distribution is given by

$$
p(\mathbf{x})=\frac{1}{N}\sum_{i=1}^{N}\delta(\mathbf{x}-\mathbf{x}_{i})
$$