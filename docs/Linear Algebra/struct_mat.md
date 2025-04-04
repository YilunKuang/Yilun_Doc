---
layout: default
title: Structured Matrices
parent: Linear Algebra
---

# Structured Matrices


## Block Diagonal Matrices $$\mathcal{BD}^{(m,n)}$$
A matrix $$\mathbf{R}\in\mathbb{R}^{n\times n}$$ is a block diagonal matrix if it can be written as 

$$
\begin{align}
    \mathbf{R}=\bigoplus_{i=1}^{\frac{n}{m}}\mathbf{R}_{i}=\begin{pmatrix}
        \mathbf{R}_{1} & \mathbf{0} & \cdots & \mathbf{0} \\
        \mathbf{0} & \mathbf{R}_{2} & \cdots & \mathbf{0} \\
        \vdots & \vdots & \ddots & \vdots \\
        \mathbf{0} & \mathbf{0} & \cdots & \mathbf{R}_{\frac{n}{m}}
    \end{pmatrix}
\end{align}
$$

where each $$\mathbf{R}_i\in\mathbb{R}^{m\times m}$$ is a dense matrix. We call the class of all block diagonal matrices in this form $$\mathcal{BD}^{(m,n)}$$.

## Diagonal Block Matrices $$\mathcal{DB}^{(m,n)}$$
A matrix $$\mathbf{L}\in\mathbb{R}^{n\times n}$$ is a diagonal block matrix if it can be written as 

$$
\begin{align}
    \mathbf{L}=\begin{pmatrix}
        \mathbf{D}_{1,1} & \mathbf{D}_{1,2} & \cdots & \mathbf{D}_{1,\frac{n}{m}} \\
        \mathbf{D}_{2,1} & \mathbf{D}_{2,2} & \cdots & \mathbf{D}_{2,\frac{n}{m}} \\
        \vdots & \vdots & \ddots & \vdots \\
        \mathbf{D}_{\frac{n}{m},1} & \mathbf{D}_{\frac{n}{m},2} & \cdots & \mathbf{D}_{\frac{n}{m},\frac{n}{m}}
    \end{pmatrix}
\end{align}
$$

where each $$\mathbf{D}_{i,j}\in\mathbb{R}^{m\times m}$$ is a diagonal matrix. We call the class of all diagonal block matrices in this form $$\mathcal{DB}^{(m,n)}$$.

## Kronecker Product

### Standard Definition

Let $$\mathbf{A}\in\mathbb{R}^{m_1\times m_2}, \mathbf{B}\in\mathbb{R}^{m_3\times m_4}$$, then the Kronecker product $$\mathbf{A}\otimes\mathbf{B}\in\mathbb{R}^{m_1 m_3\times m_2 m_4}$$ is a block matrix. 

$$
\begin{align}
\mathbf{A} \otimes \mathbf{B} =
\begin{bmatrix}
a_{11} \mathbf{B} & a_{12} \mathbf{B} & \cdots & a_{1m} \mathbf{B} \\
a_{21} \mathbf{B} & a_{22} \mathbf{B} & \cdots & a_{2m} \mathbf{B} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m 1} \mathbf{B} & a_{m 2} \mathbf{B} & \cdots & a_{mm} \mathbf{B}
\end{bmatrix}.
\end{align}
$$

where $$a_{ij}$$ is the $$ij$$-th entry of the matrix $$\mathbf{A}$$. Assume $$n=m^2$$ and $$m_1=m_2=m_3=m_4=m$$. The efficient matrix-vector multiply $$\mathbf{y}=\mathbf{W}\mathbf{x}$$ for $$\mathbf{x}, \mathbf{y}\in\mathbb{R}^{n}$$ requires reshaping $$\mathbf{x}, \mathbf{y}$$ to have shapes $$\mathbf{x}\in\mathbb{R}^{m\times m}, \mathbf{y}\in\mathbb{R}^{m\times m}$$ and we have

$$
\begin{align}
\mathbf{y}_{\alpha\beta}=\sum_{\gamma}\mathbf{A}_{\alpha\gamma}\sum_{\delta}\mathbf{B}_{\beta\delta}\mathbf{x}_{\gamma\delta}
\end{align}
$$

with the following einsum notation:

```python
def kronecker_einsum(A, B):
    return rearrange(np.einsum('ac,bd->abcd', A, B), 
        'a b c d -> (a b) (c d)')
```

### Kronecker Product as $$\mathcal{DB}^{(m,n)}$$$$\mathcal{BD}^{(m,n)}$$

We can equivalently write the matrix-vector multiply $$\mathbf{y}=(\mathbf{A}\otimes\mathbf{B})\mathbf{x}$$ as $$\mathbf{y}=\mathbf{L}\mathbf{R}\mathbf{x}$$ where 


- $$\mathbf{L}\in\mathcal{DB}^{(m,n)}$$ is a diagonal block matrix where each block $$\mathbf{D}_{i,j}\in\mathbb{R}^{m\times m}$$ is a scaler times identity $$a_{ij}\mathbf{I}_{m}$$.
- $$\mathbf{R}\in\mathcal{BD}^{(m,n)}$$ is a block diagonal matrix with each diagonal block being $$\mathbf{B}\in\mathbb{R}^{m\times m}$$ such that $$\mathbf{R}=\bigoplus_{i=1}^{m}\mathbf{B}$$



Then we have the matrix-vector multiply for $$\mathbf{y}=\mathbf{L}\mathbf{R}\mathbf{x}$$:

$$
\begin{align}
\begin{bmatrix}
    \vert \\
    \mathbf{y} \\
    \vert \\
\end{bmatrix} &=
\begin{bmatrix}
a_{11} \mathbf{I}_{m} & a_{12} \mathbf{I}_{m} & \cdots & a_{1m} \mathbf{I}_{m} \\
a_{21} \mathbf{I}_{m} & a_{22} \mathbf{I}_{m} & \cdots & a_{2m} \mathbf{I}_{m} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m 1} \mathbf{I}_{m} & a_{m 2} \mathbf{I}_{m} & \cdots & a_{m m} \mathbf{I}_{m}
\end{bmatrix}
\begin{bmatrix}
\mathbf{B} & \mathbf{0} & \cdots & \mathbf{0} \\
\mathbf{0} & \mathbf{B} & \cdots & \mathbf{0} \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{0} & \mathbf{0} & \cdots & \mathbf{B}
\end{bmatrix}
\begin{bmatrix}
    \vert \\
    \mathbf{x} \\
    \vert \\
\end{bmatrix}
\end{align}
$$

Thus Kronecker product can be interpreted as the product of a diagonal block matrix and a block diagonal matrix. 


## Butterfly Matrices

### Butterfly Factor

A **butterfly factor** $$\mathbf{B}_i\in\mathcal{BF}^{(k,k)}$$ of even size $$(k, k)$$ has the form

$$
\begin{align}
    \mathbf{B}_i &= \begin{pmatrix}
        \mathbf{D}_1 & \mathbf{D}_2 \\
        \mathbf{D}_3 & \mathbf{D}_4 \\
    \end{pmatrix}
\end{align}
$$

where $$\mathbf{D}_i$$ is a diagonal matrix of size $$(\frac{k}{2},\frac{k}{2})$$.

### Butterfly Factor Matrix

A **butterfly factor matrix** $$\mathbf{B}_{n,k}\in\mathcal{BF}^{(n,k)}$$ of size $$(n, n)$$ is a block diagonal matrix with $$n/k$$ blocks where each block is a butterfly factor of size $$(k, k)$$:

$$
\begin{align}
    \mathbf{B}_{n,k}=\text{BlockDiag}\bigg(\mathbf{B}_1, \mathbf{B}_2 \cdots, \mathbf{B}_{\frac{n}{k}}\bigg)=\bigoplus_{i=1}^{\frac{n}{k}}\mathbf{B}_i
\end{align}
$$

### Butterfly Matrix

A **butterfly matrix** of size $$(n,n)$$ for $$n=2^s$$ is a matrix $$\mathbf{M}\in\mathcal{B}^{(n)}$$ defined as a product of butterfly factor matrices:

$$
\begin{align}
    \mathbf{M} &= \mathbf{B}_{n,n}\mathbf{B}_{n,n/2}\cdots\mathbf{B}_{n,2}
\end{align}
$$

This is a product of $$\log_2 n$$ many block diagonal matrices.

### Butterfly Matrix as $$\mathcal{DB}^{(m,n)}$$$$\mathcal{BD}^{(m,n)}$$


**Theorem.** A butterfly matrix $$\mathbf{M}\in\mathcal{B}^{(n)}$$ can be represent as $$\mathbf{M}=\mathbf{L}\mathbf{R}$$ for $$\mathbf{L}\in\mathcal{DB}^{(m,n)}$$ and $$\mathbf{R}\in\mathcal{BD}^{(m,n)}$$.


## Monarch Matrices

### Definition

A **monarch matrix** of size $$(n,n)$$ for $$n=m^2$$ is a matrix $$\mathbf{M}\in\mathcal{M}^{(n)}$$ with the form:

$$
\begin{align}
    \mathbf{M}=\mathbf{P}\mathbf{L}\mathbf{P}^\top\mathbf{R}
\end{align}
$$

where $$\mathbf{L}=\bigoplus_{i=1}^{m}\mathbf{L}_i$$ and $$\mathbf{R}=\bigoplus_{i=1}^{m}\mathbf{R}_i$$ are block-diagonal matrices for $$\mathbf{L}_i,\mathbf{R}_i\in\mathbb{R}^{m\times m}$$. The permutation matrix $$\mathbf{P}$$ maps $$[x_1, \cdots, x_n]$$ to $$[x_1,x_{m+1},\cdots,\\x_{(m-1)m+1},x_2,x_{2+m},\cdots,x_{(m-1)m+2},\cdots,x_{m},x_{2m},\cdots,x_n]$$.

### Relationship with Butterfly Matrices

**Lemma.** If $$\mathbf{M}\in\mathcal{B}^{(n)}$$, then $$\mathbf{M}\in\mathcal{M}^{(n)}$$.


**Proof.** Since $$\mathbf{M}\in\mathcal{B}^{(n)}$$, we can write $$\mathbf{M}=\mathbf{B}_{n,n}\mathbf{B}_{n,n/2}\cdots\mathbf{B}_{n,2}$$. Let the product of the first $$\frac{\log_2 n}{2}$$ butterfly factor matrices be $$\mathbf{L}'$$ and the product of the last $$\frac{\log_2 n}{2}$$ butterfly factor matrices be $$\mathbf{R}$$. Thus $$\mathbf{M}=\mathbf{L}'\mathbf{R}$$. Here $$\mathbf{R}=\bigoplus_{i=1}^{m}\mathbf{R}_i$$ for $$\mathbf{R}_i\in\mathbb{R}^{m\times m}$$. Notice that we can write $$\mathbf{L}'$$ as:

$$
\begin{align}
    \mathbf{L}'=\begin{pmatrix}
        \mathbf{D}_{11} & \cdots & \mathbf{D}_{1m}\\
        \vdots & \ddots & \vdots \\
        \mathbf{D}_{m1} & \cdots & \mathbf{D}_{mm}
    \end{pmatrix}
\end{align}
$$

where each $$\mathbf{D}_{ij}$$ is a diagonal matrix with size $$(m, m)$$. It can be verified that 

$$
\begin{align}
    \mathbf{L}=\mathbf{P}\mathbf{L}'\mathbf{P}^\top
\end{align}
$$

where $$\mathbf{L}=\bigoplus_{i=1}^{m}\mathbf{L}_{i}$$ is a block diagonal matrix for $$\mathbf{L}_i\in\mathbb{R}^{m\times m}$$. Thus $$\mathbf{M}\in\mathcal{M}^{(n)}$$.

### Relationship with Kronecker Product

It's obvious that Monarch matrices generalize Kronecker products since both of them can be written as products between a diagonal block matrix and a block diagonal matrix. 


