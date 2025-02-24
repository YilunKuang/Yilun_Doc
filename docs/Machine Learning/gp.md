---
layout: default
title: Gaussian Process
parent: Machine Learning
---

# Gaussian Process

## Gaussian Processes for Machine Learning Chapter 2 Summary

### 2.1 Weight-Space View

For a linear regression model $$y=f(\mathbf{x})+\epsilon=\mathbf{x}^\top\mathbf{w}+\epsilon$$, where $$\epsilon\sim\mathcal{N}(0,\sigma_n^2)$$ and $$\mathbf{w}\sim\mathcal{N}(\mathbf{0},\Sigma_p)$$, we have the posterior distribution $$p(\mathbf{w}\vert X,y)\sim\mathcal{N}(\overline{\mathbf{w}}=\frac{1}{\sigma_n^2}A^{-1}Xy, A^{-1})$$, where $$A=\sigma_n^{-2}XX^\top+\Sigma_p^{-1}$$. We also have the predictive distribution $$p(f(\mathbf{x}_*)\vert \mathbf{x}_*,X,\mathbf{y})=\mathcal{N}(\frac{1}{\sigma_n^2}\mathbf{x}_*^\top A^{-1}X\mathbf{y},\mathbf{x}_*^\top A^{-1}\mathbf{x}_*)$$.

If we let $$f(\mathbf{x})=\phi(\mathbf{x})^\top\mathbf{w}$$, then the predictive distribution is given by $$f(\mathbf{x}_*)\vert \mathbf{x}_*,X,\mathbf{y}\sim\mathcal{N}(\mathbf{\phi}_*^\top\Sigma_p\Phi(K+\sigma_n^2I)^{-1}\mathbf{y},\mathbf{\phi}_*^\top\Sigma_p\mathbf{\phi}_*-\mathbf{\phi}_*^\top\Sigma_p\Phi(K+\sigma_n^2I)^{-1}\Phi^\top\Sigma_p\mathbf{\phi}_*)$$.

### 2.2 Function-Space View

Consider the Bayesian linear regression model $$f(\mathbf{x})=\phi(\mathbf{x})^\top\mathbf{w}$$ with prior $$\mathbf{w}\sim\mathcal{N}(\mathbf{0},\Sigma_p)$$. Then we have the mean and covariance function

$$
m(\mathbf{x})=\mathbb{E}[f(\mathbf{x})]=\phi(\mathbf{x})^\top\mathbb{E}[\mathbf{w}]=0\\
$$
$$
k(\mathbf{x},\mathbf{x'})=\mathbb{E}[(f(\mathbf{x})-m(\mathbf{x}))(f(\mathbf{x'})-m(\mathbf{x}'))]\\
=\mathbb{E}[f(\mathbf{x})f(\mathbf{x'})]=\phi(\mathbf{x})^\top\Sigma_p\phi(\mathbf{x}')
$$

we say $$f(\mathbf{x})\sim \mathcal{GP}(m(\mathbf{x}),k(\mathbf{x},\mathbf{x'}))$$ as a Gaussian process, completely determined by the mean function $$m(\mathbf{x})$$ and the covariance function $$k(\mathbf{x},\mathbf{x'})$$. 

Choose the RBF kernel $$k(\mathbf{x}_p,\mathbf{x}_q)=\exp(-\frac{1}{2}\vert \mathbf{x}_p-\mathbf{x}_q\vert ^2)$$. Consider the model $$y=f(\mathbf{x})+\epsilon$$. Then we have the covariance function $$\text{cov}(y_p,y_q)=k(\mathbf{x}_p,\mathbf{x}_q)+\sigma_n^2\delta_{pq}$$. The predictive equation for Gaussian process regression is given by $$\mathbf{f}_*\vert X,\mathbf{y},X_*\sim\mathcal{N}(\overline{\mathbf{f}}_*,\text{cov}(\mathbf{f}_*))$$, where

$$
\overline{f}_*=\mathbf{k}_*^\top(K+\sigma_n^2I)^{-1}\mathbf{y}
$$

$$
\mathbb{V}[f_*]=k(\mathbf{x}_*,\mathbf{x}_*)-\mathbf{k}_*^\top(K+\sigma_n^2I)^{-1}\mathbf{k}_*
$$


### 2.3 Varying the Hyperparameters

The squared-exponential covariance function in one dimension has the following form:

$$
k_y(x_p,x_q)=\sigma_f^2\exp(-\frac{1}{2\ell^2}(x_p-x_q)^2)+\sigma_n^2\delta_{pq}
$$

The length-scale hyperparameter $$\ell$$ affects variances in the output prediction. Hyperparameters $$\sigma_f^2$$ and $$\sigma_n^2$$ can also be set by optimizing marginal likelihood.


### 2.4 Decision Theory for Regression

Given the predictive distribution, we would like to compute a point estimation for decision. We can choose $$y$$ such that 

$$
y_\text{optimal}\vert \mathbf{x}_*=\text{argmin}_{y_{\text{guess}}}\int\mathcal{L}(y_*,y_{\text{guess}})p(y_*\vert \mathbf{x}_*,\mathcal{D})dy_{*}
$$

### 2.5 An Example Application

See textbook.

## Gaussian Processes for Machine Learning Chapter 4 Summary


### 4.0 Introduction
“At the heart of every Gaussian process model - controlling all the modeling power - is a covariance kernel” (Andrew Wilson’s Thesis, p. 39). A covariance kernel $$k(\mathbf{x},\mathbf{x'})$$ encodes inductive biases into our model. Indeed, if a neural network / brain $$f\sim\mathcal{GP}(m(\mathbf{x}),k(\mathbf{x},\mathbf{x'}))$$ with SSL representations, then we would expect the covariance kernel to encodes notions of similarity in the representations.

### 4.1 Preliminaries

A stationary covariance function is a function of $$\tau:=\mathbf{x}-\mathbf{x}'$$ and an isotropic covariance function is a function of $$\vert \mathbf{x}-\mathbf{x}'\vert $$. The kernel arises in the theory of integral operators: $$(T_kf)(\mathbf{x})=\int_{\mathcal{X}}k(\mathbf{x},\mathbf{x}')f(\mathbf{x}')d\mu(\mathbf{x}')$$.

#### 4.1.1 Mean Square Continuity and Differentiability

For a sequence of points $$\{\mathbf{x}_1,\mathbf{x}_2,…\}$$ which converges to a fixed point $$\mathbf{x}_*\in\mathbb{R}^D$$, a stochastic process $$f(\mathbf{x})$$ is said to be continuous in mean squared at $$\mathbf{x}_*$$ if $$\lim_{k\to\infty}\mathbb{E}[\vert f(\mathbf{x}_k)-f(\mathbf{x}_*)\vert ^2]= 0$$.

### 4.2 Examples of Covariance Functions

#### 4.2.1 Stationary Covariance Functions

By Bochner’s Theorem, the spectral density / power spectrum and the stationary kernel $$k$$ are Fourier duals

$$
k(\tau)=\int S(s)e^{2\pi is^\top \tau}ds\\
S(s)=\int k(\tau)e^{-2\pi is^\top \tau}d\tau
$$

So the spectral density determines the inductive biases we’re encoding in our models. Some example isotropic kernels include the Squared Exponential (SE) kernel (also called the RBF kernel), Matérn class of covariance functions, OU process, Rational Quadratic Kernel etc.

#### 4.2.2 Dot Product Covariance Functions

Kernels in the form of $$k(\mathbf{x},\mathbf{x}')=\sigma_0^2+\mathbf{x}\cdot \mathbf{x}'$$ are dot product covariance functions.

#### 4.2.3 Other Non-Stationary Covariance Functions

NNGP kernel for an infinitely-wide neural network is given by

$$
\mathbb{E}_{\mathbf{w}}[f(\mathbf{x})f(\mathbf{x}')]=\sigma_b^2+N_H\sigma_v^2\mathbb{E}_{\mathbf{u}}[h(\mathbf{x};\mathbf{u})h(\mathbf{x}';\mathbf{u})]
$$

### 4.3 Eigenfunction Analysis of Kernels

A function $$\phi(\cdot)$$ is called an eigenfunction of the kernel $$k$$ with eigenvalue $$\lambda$$ if $$\int k(\mathbf{x},\mathbf{x}')\phi(\mathbf{x})d\mu(\mathbf{x})=\lambda\phi(\mathbf{x}')$$. 

By the Mercer’s Theorem, we have the following decomposition of kernels into components in infinite basis: $$k(\mathbf{x},\mathbf{x}')=\sum_{i=1}^{\infty}\lambda_i\phi_i(\mathbf{x})\phi_i^*(\mathbf{x}')$$

The eigenfunctions can be approximated using common methods in numerical analysis regarding the eigenvalue problems.

### 4.4 Kernels for Non-Vectorial Inputs

Kernels can also be defined over structured object. String kernels and Fisher kernels are examples. 
