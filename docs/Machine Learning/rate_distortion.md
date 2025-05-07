---
layout: default
title: Rate Distortion
parent: Machine Learning
---


# Maximal Coding Rate Reduction Principle


### Rate Distortion, Coding Length Function, and Coding Length Function of Segmented Data **[MDHW07]**

**The** **Rate-Distortion Function**

Consider a set of encoding representations $$W=(w_1,...,w_m)\in\mathbb{R}^{n\times m},w_i\in\mathbb{R}^n\space\forall\space i=\{1,...,m\}$$, where $$n$$ is the number of features and $$m$$ is the number of samples. 

The set of encoding representations $$W$$ is assumed to be zero mean for the sake of simplicity, i.e. $$\mu=\frac{1}{m}\sum_{i=1}^{m}w_i=0$$. 

For every $$w_i$$, an approximation $$\hat{w}_i$$ of $$w_i$$ must satisfy $$\mathbb{E}[\|\|w_i-\hat{w}_i\|\|^2]\leq\epsilon^2$$. In other words, the encoding error of every vector $$w_i$$ is bounded by $$\epsilon^2\implies$$the encoding error of every entry of $$w_i$$ is $$\epsilon^2/n$$. Without the loss of generality, $$\hat{w}_i=w_i+z_i,\space z_i\sim\mathcal{N}(0,\frac{\epsilon^2}{n}I)$$.

For a given set of representations $$W=(w_1,...,w_m)$$, the set of approximations $$\hat{W}=(\hat{w}_1,...,\hat{w}_m)$$ is now a collection of spheres. Optimal coding is now equivalent to optimal sphere packing subjected to the encoding error $$\epsilon^2$$ constraints. Notice that the covariance matrix of the vectors $$\hat{w}_i$$ is

$$
\hat{\Sigma}=\mathbb{E}\bigg[\frac{1}{m}\sum_{i=1}^{m}\hat{w}_i\hat{w}_i^\top\bigg]=\frac{\epsilon^2}{n}I+\frac{1}{m}WW^\top
$$

The volume of $$\hat{W}$$ is proportional to the square root of the determinant of the covariance matrix

$$
\text{vol}(\hat{W})\propto\sqrt{\det\bigg(\frac{\epsilon^2}{n}I+\frac{1}{m}WW^\top\bigg)}
$$

Since $$\det(X)=\prod_i\lambda_i(X)$$, the above volume terms is the product of the eigenvalues of the covariance matrix, which has a nice interpretation in terms of PCA. In other words, the volume measure $$\text{vol}(\cdot)$$ is the product of the magnitude of all principle directions of the encoding representation.

Similarly, the volume of an individual sphere is defined to be

$$
\text{vol}(z)\propto\sqrt{\det\bigg(\frac{\epsilon^2}{n}I\bigg)}
$$

The total number of spheres we can packed into the region spanned by $$\hat{W}$$ is then given by the ratio

$$
\# \text{ of spheres}=\frac{\text{vol}(\hat{W})}{\text{vol}(z)},
$$

where each sphere is non-overlapping. So now to know where each vector $$w_i$$ is, we just need to label all the spheres. If we use binary numbers to label the spheres, then the number of bits needed is 

$$
R(W)
=\log_2(\# \text{ of spheres})
=\frac{1}{2}\log_2\Bigg(\frac{\det\big(\frac{\epsilon^2}{n}I+\frac{1}{m}WW^\top\big)}{\det\big(\frac{\epsilon^2}{n}I\big)}\Bigg)
\\=\frac{1}{2}\log_2\det\bigg(I+\frac{n}{m\epsilon^2}WW^\top\bigg)
$$

This is basically the rate distortion function.

**The Coding Length Function**

The coding length function is given by

$$
L(W)=(m+n)R(W)=\frac{m+n}{2}\log_2\det\bigg(I+\frac{n}{m\epsilon^2}WW^\top\bigg)
$$

**Coding Length for Segmented Data**

Given a set of $$m$$ vectors $$W=(w_1,w_2,...,w_m)$$, $$w_i\in\mathbb{R}^n$$, we define a non-overlapping partition into $$k$$ subsets $$W=W_1\cup W_2\cup...\cup W_k$$. Then the total number of bits to encode the data with segmentations is given by

$$
L^s(W_1,...,W_k)=\sum_{i=1}^{k}L(W_i)+ |W_i|(-\log_2\frac{|W_i|}{m}),
$$

where $$\sum_{i=1}^kL(W_i)$$ is the coding length for all the subgroup $$W_i$$ and $$\sum_{i=1}^k\| W_i\|(-\log_2\frac{\| W_i\|}{m})$$ is the number of bits needed to encode the membership of the $$m$$ samples in the $$k$$ groups. 

**The** **Rate-Distortion Function for Segmented Data** 

Consider the probabilistic segmentation settings, where each vector $$w_i$$ is assigned to group $$j$$ with a probability $$\pi_{ij}\in[0,1]$$, with $$\sum_{j=1}^{k}\pi_{ij}=1\space\forall\space i=1,...,m$$.

We denote the matrix $$\Pi_j$$ that collects the membership of the $$m$$ vectors in group $$j$$ as:

$$
\Pi_j=
\begin{bmatrix}
\pi_{1j} & 0 & \ldots & 0\\
0 & \pi_{2j} & \ldots & 0\\
\vdots & \vdots &\ddots & \vdots\\
0 & 0 & \ldots &\pi_{mj}
\end{bmatrix}\in\mathbb{R}^{m\times m}.
$$

These matrices satisfy the constraint $$\sum_{j=1}^{k}\Pi_j=I_{m\times m}$$. Notice that the $$j$$-th group has an expected number of $$\text{tr}(\Pi_j)$$ vectors, and the expected covariance is given by $$\frac{1}{\text{tr}(\Pi_j)}W\Pi_jW^\top$$. The coding length function for segmented data is then given by

$$
L^s(W,\Pi)=\sum_{j=1}^{k}\frac{\text{tr}(\Pi_j)+n}{2}\log_2\det\bigg(I+\frac{n}{\text{tr}(\Pi_j)\epsilon^2}W\Pi_jW^\top\bigg)+\\\text{tr}(\Pi_j)\bigg(-\log_2\frac{\text{tr}(\Pi_j)}{m}\bigg)
$$

The expected number of bits to encode each vector is then

$$
\begin{align}
    
R^s(W,\Pi)&=\frac{1}{m}L^s(W,\Pi)
\\&=\sum_{j=1}^{k}\frac{\text{tr}(\Pi_j)+n}{2m}\log_2\det\bigg(I+\frac{n}{\text{tr}(\Pi_j)\epsilon^2}W\Pi_jW^\top\bigg)+\frac{\text{tr}(\Pi_j)}{m}\bigg(-\log_2\frac{\text{tr}(\Pi_j)}{m}\bigg)
\\&=\sum_{j=1}^{k}\frac{\text{tr}(\Pi_j)}{2m}\log_2\det\bigg(I+\frac{n}{\text{tr}(\Pi_j)\epsilon^2}W\Pi_jW^\top\bigg)+\frac{n}{2m}\log_2\det\bigg(I+\frac{n}{\text{tr}(\Pi_j)\epsilon^2}W\Pi_jW^\top\bigg)+\frac{\text{tr}(\Pi_j)}{m}\bigg(-\log_2\frac{\text{tr}(\Pi_j)}{m}\bigg)
\\&=\sum_{j=1}^{k}\frac{\text{tr}(\Pi_j)}{m}\frac{1}{2}\log_2\det\bigg(I+\frac{n}{\text{tr}(\Pi_j)\epsilon^2}W\Pi_jW^\top\bigg)+\frac{n}{m}R(W_j)+\frac{\text{tr}(\Pi_j)}{m}\bigg(-\log_2\frac{\text{tr}(\Pi_j)}{m}\bigg)
\\&=\sum_{j=1}^{k}\frac{\text{tr}(\Pi_j)}{m}\Bigg[\frac{1}{2}\log_2\det\bigg(I+\frac{n}{\text{tr}(\Pi_j)\epsilon^2}W\Pi_jW^\top\bigg)-\log_2\frac{\text{tr}(\Pi_j)}{m}\Bigg]+\frac{n}{m}R(W_j)
\\&=\sum_{j=1}^{k}\frac{\text{tr}(\Pi_j)}{m}\Bigg[R(W_j)-\log_2\frac{\text{tr}(\Pi_j)}{m}\Bigg]+\frac{n}{m}R(W_j)
\end{align}
$$

The asymptotic limit of the above function as $$m\to\infty$$ is given by

$$
R^{s,\infty}(W,\Pi)=\sum_{j=1}^{k}\frac{\text{tr}(\Pi_j)}{2m}\log_2\det\bigg(I+\frac{n}{\epsilon^2\text{tr}(\Pi_j)}W\Pi_jW^\top\bigg)\\-\frac{\text{tr}(\Pi_j)}{m}\log_2\bigg(\frac{\text{tr}(\Pi_j)}{m}\bigg)
$$

### Principle of Maximal Coding Rate Reduction

$$
\max_{\theta,\Pi}\Delta R(Z(\theta),\Pi,\epsilon)=R(Z(\theta),\epsilon)-R^c(Z(\theta),\epsilon|\Pi)
\\
\text{s.t. }||Z_j(\theta)||_F^2=m_j,\Pi\in\Omega
$$


```
@inproceedings{NEURIPS2020_6ad4174e,
 author = {Yu, Yaodong and Chan, Kwan Ho Ryan and You, Chong and Song, Chaobing and Ma, Yi},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
 pages = {9422--9434},
 publisher = {Curran Associates, Inc.},
 title = {Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction},
 url = {https://proceedings.neurips.cc/paper/2020/file/6ad4174eba19ecb5fed17411a34ff5e6-Paper.pdf},
 volume = {33},
 year = {2020}
}
```

```
Y. Ma, H. Derksen, W. Hong and J. Wright, "Segmentation of Multivariate Mixed Data via Lossy Data Coding and Compression," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 29, no. 9, pp. 1546-1562, Sept. 2007, doi: 10.1109/TPAMI.2007.1085.
```
