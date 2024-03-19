---
layout: default
title: Iterative Methods
parent: Numerical Methods
---
# Iterative Methods

Suppose we want to solve a linear system $$A\mathbf{x}=\mathbf{b}$$ where $$A\in\mathbb{R}^{n\times n}$$. Let $$Q$$ be an invertible matrix such that

$$
\begin{align}
A\mathbf{x}&=\mathbf{b} \\
\iff Q^{-1}(\mathbf{b}-A\mathbf{x})&=0 \\
\iff Q^{-1}\mathbf{b}-Q^{-1}A\mathbf{x}&=0 \\
\iff \mathbf{x}+Q^{-1}\mathbf{b}-Q^{-1}A\mathbf{x}&=\mathbf{x} \\
\iff (I-Q^{-1}A)\mathbf{x}+Q^{-1}\mathbf{b}&=\mathbf{x} \\
\iff G\mathbf{x}+\mathbf{c}&=\mathbf{x} \\
\end{align}
$$

where $$G=I-Q^{-1}A$$ and $$Q^{-1}\mathbf{b}=\mathbf{c}$$. This is equivalent to a fixed-point iteration where the stationary point is $$\mathbf{x}=A^{-1}\mathbf{b}$$:

$$
\begin{align}
\mathbf{x}_{k+1}&=G\mathbf{x}_{k}+\mathbf{c} \\
\end{align}
$$

We can decompose the matrix $$A$$ into lower-triangular $$L$$, diagonal $$D$$, and upper-triangular $$U$$ components such that $$A=L+D+U$$. 

## Gauss-Seidel

If we pick $$Q=D+L$$, we have the Gauss-Seidel method that converges for any initial guess $$\mathbf{x}_{0}$$ if $$A$$ is symmetric positive definite: 

$$
\begin{align}
\mathbf{x}_{k+1}&=(I-(D+L)^{-1}A)\mathbf{x}_{k}+(D+L)^{-1}\mathbf{b} \\
\mathbf{x}_{k+1}&=(I-(D+L)^{-1}(D+L+U))\mathbf{x}_{k}+(D+L)^{-1}\mathbf{b} \\
\mathbf{x}_{k+1}&=-(D+L)^{-1}U\mathbf{x}_{k}+(D+L)^{-1}\mathbf{b} \\
\mathbf{x}_{k+1}&=(D+L)^{-1}(\mathbf{b}-U\mathbf{x}_{k}) \\
\color{lightskyblue}{(D+L)}\color{yellowgreen}{\mathbf{x}_{k+1}}&=\color{fuchsia}{(\mathbf{b}-U\mathbf{x}_{k})} \\
\end{align}
$$

Notice that $$(D+L)$$ is an upper triangular matrix so we can use forward substitution to solve for $$\mathbf{x}_{k+1}$$:

$$
\begin{align}
\color{yellowgreen}{\mathbf{x}_{k+1}[i]}&=\color{lightskyblue}{\frac{1}{A[i][i]}}\bigg(\color{fuchsia}{\mathbf{b}[i]-\sum_{j=i+1}^{n}A[i][j]\mathbf{x}_{k}[j]}-\color{lightskyblue}{\sum_{j=1}^{i-1}A[i][j]}\color{yellowgreen}{\mathbf{x}_{k+1}[j]}\bigg)\\
\color{yellowgreen}{\mathbf{x}_{k+1}[i]}&=\color{lightskyblue}{\frac{1}{A[i][i]}}\bigg(\color{fuchsia}{\mathbf{b}[i]-\big(\sum_{j=i}^{n}A[i][j]\mathbf{x}_{k}[j]-A[i][i]\mathbf{x}_{k}[i]\big)}-\color{lightskyblue}{\sum_{j=1}^{i-1}A[i][j]}\color{yellowgreen}{\mathbf{x}_{k+1}[j]}\bigg)\\
\color{yellowgreen}{\mathbf{x}_{k+1}[i]}&=\color{lightskyblue}{\frac{1}{A[i][i]}}\bigg(\color{fuchsia}{\mathbf{b}[i]-\sum_{j=i}^{n}A[i][j]\mathbf{x}_{k}[j]+A[i][i]\mathbf{x}_{k}[i]}-\color{lightskyblue}{\sum_{j=1}^{i-1}A[i][j]}\color{yellowgreen}{\mathbf{x}_{k+1}[j]}\bigg)\\
\color{yellowgreen}{\mathbf{x}_{k+1}[i]}&=\frac{\color{fuchsia}{A[i][i]\mathbf{x}_{k}[i]}}{\color{lightskyblue}{A[i][i]}}+\color{lightskyblue}{\frac{1}{A[i][i]}}\bigg(\color{fuchsia}{\mathbf{b}[i]-\sum_{j=i}^{n}A[i][j]\mathbf{x}_{k}[j]}-\color{lightskyblue}{\sum_{j=1}^{i-1}A[i][j]}\color{yellowgreen}{\mathbf{x}_{k+1}[j]}\bigg)\\
\color{yellowgreen}{\mathbf{x}_{k+1}[i]}&=\color{fuchsia}{\mathbf{x}_{k}[i]}+\color{lightskyblue}{\frac{1}{A[i][i]}}\bigg(\color{fuchsia}{\mathbf{b}[i]-\sum_{j=i}^{n}A[i][j]\mathbf{x}_{k}[j]}-\color{lightskyblue}{\sum_{j=1}^{i-1}A[i][j]}\color{yellowgreen}{\mathbf{x}_{k+1}[j]}\bigg)\\
\end{align}
$$

Thus we arrive at the Gauss-Seidel fixed-point iteration. Notice that this iteration only requires a single row from the matrix $$A$$ during the update for position $$i$$. This means when the matrix $$A$$ is large, we don't need to store the entire dense matrix $$A$$. Since each position $$i$$ can be computed independently, the Gauss-Seidel iteration can also be massively parallelized on modern GPUs for each position $$i$$. 