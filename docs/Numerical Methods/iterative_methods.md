---
layout: default
title: Iterative Methods
parent: Numerical Methods
---
# Iterative Methods

## Fixed-Point Iteration
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

The convergence criteria of the sequence $$(\mathbf{x}_k)_{k\in\mathbb{N}\cup\{0\}}$$ is given by the following theorem: 

<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">

<strong>Theorem:</strong> <em>The fixed-point method \( \mathbf{x}_{k+1}=G\mathbf{x}_{k}+\mathbf{c} \) with an invertible \( G \) converges for any initial point \( \mathbf{x}_{0} \) if and only if the spectral radius of \( G \) is smaller than \( 1 \):</em>

\[ \rho(G)<1, \]

<em>where \( \rho(G):=\max_{j}\lvert \lambda_{j}\rvert \) is the largest eigenvalue of \( G \) in absolute values.</em>

</div>

## Different Iterative Methods

Different types of fixed point iteration correspond to difference choices of the matrix $$Q$$. In the following section we decompose the matrix $$A$$ into lower-triangular $$L$$, diagonal $$D$$, and upper-triangular $$U$$ components such that $$A=L+D+U$$. 


### Dense Method
In the extreme case if we simply choose $$Q=A$$, then the fixed-point iteration gives

$$
\begin{align}
\mathbf{x}_{k+1}&=(I-A^{-1}A)\mathbf{x}_{k}+A^{-1}\mathbf{b} \\
\iff\mathbf{x}_{k+1}&=\mathbf{0}+\mathbf{x} \\
\iff\mathbf{x}_{k+1}&=\mathbf{x} \\
\end{align}
$$

We arrive at the solution in one step. This is equivalent to dense methods such as LU factorizations or QR decompositions where we simply compute $$A^{-1}$$.

### Richardson
The other extreme case is choosing $$Q=I$$. Then we have the Richardson method:


$$
\begin{align}
\mathbf{x}_{k+1}&=(I-A)\mathbf{x}_{k}+\mathbf{b} \\
\end{align}
$$

Since we have invested zero costs in finding a proper preconditioner $$Q$$, we can only expect slow convergence or even divergence. Algorithmically, we can rewrite the update as:

$$
\begin{align}
\mathbf{x}_{k+1}&=\mathbf{x}_{k}+(\mathbf{b}-A\mathbf{x}_{k}) \\
\mathbf{x}_{k+1}[i]&=\mathbf{x}_{k}[i]+\bigg(\mathbf{b}[i]-\sum_{i=1}^{n}A[i][j]\mathbf{x}_{k}[j]\bigg) \\
\end{align}
$$

where $$\mathbf{v}[i]$$ is the $$i$$-th entry of the vector $$\mathbf{v}\in\mathbb{R}^{n}$$. Notice that the Richardon method updates the solution in the direction of the residual $$\mathbf{r}_{k}:=\mathbf{b}-A\mathbf{x}_k=A(\mathbf{x}-\mathbf{x}_k)=A\mathbf{e}_k$$ where $$\mathbf{e}_k$$ is the error. 



### Jacobi

If we pick $$Q=D$$, we have the Jacobi method that converges for any starting point $$\mathbf{x}_0$$ if $$A$$ is strictly diagonal dominant, i.e. $$\lvert A[i][i]\rvert>\sum_{j\neq i}\lvert A[i][j]\rvert$$ for $$i=1,\cdots,n$$:

$$
\begin{align}
\mathbf{x}_{k+1}&=(I-D^{-1}A)\mathbf{x}_{k}+D^{-1}\mathbf{b} \\
\end{align}
$$

We can obtain a component-wise update as follows:

$$
\begin{align}
\mathbf{x}_{k+1}&=\mathbf{x}_{k}+D^{-1}(\mathbf{b}-A\mathbf{x}_{k}) \\
\mathbf{x}_{k+1}[i]&=\mathbf{x}_{k}[i]+\frac{1}{A[i][i]}\bigg(\mathbf{b}[i]-\sum_{i=1}^{n}A[i][j]\mathbf{x}_{k}[j]\bigg) \\

\end{align}
$$


### Gauss-Seidel

If we pick $$Q=D+L$$, we have the Gauss-Seidel method that converges for any initial guess $$\mathbf{x}_{0}$$ if $$A$$ is symmetric positive definite (spd): 

$$
\begin{align}
\mathbf{x}_{k+1}&=(I-(D+L)^{-1}A)\mathbf{x}_{k}+(D+L)^{-1}\mathbf{b} \\
\mathbf{x}_{k+1}&=(I-(D+L)^{-1}(D+L+U))\mathbf{x}_{k}+(D+L)^{-1}\mathbf{b} \\
\mathbf{x}_{k+1}&=-(D+L)^{-1}U\mathbf{x}_{k}+(D+L)^{-1}\mathbf{b} \\
\mathbf{x}_{k+1}&=(D+L)^{-1}(\mathbf{b}-U\mathbf{x}_{k}) \\
\color{#4a90e2}{(D+L)}\color{#50b83c}{\mathbf{x}_{k+1}}&=\color{#9c6ade}{(\mathbf{b}-U\mathbf{x}_{k})} \\
\end{align}
$$

Notice that $$(D+L)$$ is an upper triangular matrix so we can use forward substitution to solve for $$\mathbf{x}_{k+1}$$:

$$
\begin{align}
\color{#50b83c}{\mathbf{x}_{k+1}[i]}&=\color{#4a90e2}{\frac{1}{A[i][i]}}\bigg(\color{#9c6ade}{\mathbf{b}[i]-\sum_{j=i+1}^{n}A[i][j]\mathbf{x}_{k}[j]}-\color{#4a90e2}{\sum_{j=1}^{i-1}A[i][j]}\color{#50b83c}{\mathbf{x}_{k+1}[j]}\bigg)\\
\color{#50b83c}{\mathbf{x}_{k+1}[i]}&=\color{#4a90e2}{\frac{1}{A[i][i]}}\bigg(\color{#9c6ade}{\mathbf{b}[i]-\big(\sum_{j=i}^{n}A[i][j]\mathbf{x}_{k}[j]-A[i][i]\mathbf{x}_{k}[i]\big)}-\color{#4a90e2}{\sum_{j=1}^{i-1}A[i][j]}\color{#50b83c}{\mathbf{x}_{k+1}[j]}\bigg)\\
\color{#50b83c}{\mathbf{x}_{k+1}[i]}&=\color{#4a90e2}{\frac{1}{A[i][i]}}\bigg(\color{#9c6ade}{\mathbf{b}[i]-\sum_{j=i}^{n}A[i][j]\mathbf{x}_{k}[j]+A[i][i]\mathbf{x}_{k}[i]}-\color{#4a90e2}{\sum_{j=1}^{i-1}A[i][j]}\color{#50b83c}{\mathbf{x}_{k+1}[j]}\bigg)\\
\color{#50b83c}{\mathbf{x}_{k+1}[i]}&=\frac{\color{#9c6ade}{A[i][i]\mathbf{x}_{k}[i]}}{\color{#4a90e2}{A[i][i]}}+\color{#4a90e2}{\frac{1}{A[i][i]}}\bigg(\color{#9c6ade}{\mathbf{b}[i]-\sum_{j=i}^{n}A[i][j]\mathbf{x}_{k}[j]}-\color{#4a90e2}{\sum_{j=1}^{i-1}A[i][j]}\color{#50b83c}{\mathbf{x}_{k+1}[j]}\bigg)\\
\color{#50b83c}{\mathbf{x}_{k+1}[i]}&=\color{#9c6ade}{\mathbf{x}_{k}[i]}+\color{#4a90e2}{\frac{1}{A[i][i]}}\bigg(\color{#9c6ade}{\mathbf{b}[i]-\sum_{j=i}^{n}A[i][j]\mathbf{x}_{k}[j]}-\color{#4a90e2}{\sum_{j=1}^{i-1}A[i][j]}\color{#50b83c}{\mathbf{x}_{k+1}[j]}\bigg)\\
\end{align}
$$

Thus we arrive at the Gauss-Seidel fixed-point iteration. Viewing from an algorithmic perspective, Gauss-Seidel improves over Jacobi by taking into account the already updated $$\mathbf{x}_{k+1}[j]$$ for $$j=1,\cdots,i-1$$ during the update of $$\mathbf{x}_{k+1}[i]$$.

<div style="background-color: #f3f5f9; padding: 10px; border-radius: 5px;">

<strong>Remark:</strong>
Notice that all of the iterative methods above only require a single row from the matrix \(A\) during the update for position \(i\). This means when the matrix \(A\) is large, we don't need to store the entire dense matrix \(A\). Since each position \(i\) can be computed independently, iterative methods can also be massively parallelized on modern GPUs for each position \(i\).

</div>




**TODO:**
- python implementation of gauss-seidel
- python / CUDA implementataion of gauss-seidel for solving linear systems in parallel


**Reference:**
- [1] NYU Numerical Method I Lecture
