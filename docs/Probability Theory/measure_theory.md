---
layout: default
title: Measure Theory
parent: Probability Theory
---
# Measure Theory

### Random Variable

A **random variable** $$X$$ is a **measurable function** from a measurable space $$(\Omega, \mathcal{F}, \mathbb{P})$$ to another measurable space $$(\mathbb{R}, \mathcal{B}(\mathbb{R}))$$, where
- $$\Omega$$ is the sample space (the set of all possible outcomes).
- $$\mathcal{F}$$ is a $$\sigma$$-algebra on $$\Omega$$ (the collection of events).
- $$\mathbb{P}$$ is a probability measure on $$(\Omega, \mathcal{F})$$
- $$\mathcal{B}(\mathbb{R})$$ is the Borel $$\sigma$$-algebra on $$\mathbb{R}$$ (the collection of measurable subsets of $$\mathbb{R}$$)

Measurability means that for every Borel set $$B\in\mathcal{B}(\mathbb{R})$$, the preimage $$X^{-1}(B)$$ is in $$\mathcal{F}$$. This ensures that we can assign probabilities to events involving $$X$$.

### Distribution of a Random Variable

The **distribution** of $$X$$, denoted $$\mathbb{P}_X$$, is a probability measure on $$\mathcal{B}(\mathbb{R})$$ that describes how the probability mass is distributed over the possible values of $$X$$. Formally, it is defined as the **pushforward measure** of $$\mathbb{P}$$ under $$X$$:

$$
\mathbb{P}_X(B)=X_*(\mathbb{P})(B)=\mathbb{P}(X^{-1}(B))
$$

for all $$B\in\mathcal{B}(\mathbb{R})$$.

### Probability Density Function (PDF)

A **probability density function (PDF)** is a function $$f_X:\mathbb{R}\to \mathbb{R}$$ that describes the distribution of a continuous random variable $$X$$ with respect to a reference measure (usually the Lebesgue measure $$\mu$$ on $$\mathbb{R}$$). The PDF satisfies:

$$
\mathbb{P}_X(B)=\int_{B} f_X(x)d\mu(x)
$$

In measure-theoretic terms, $$f_X$$ is the **Radon-Nikodym derivative** of $$\mathbb{P}_X$$ with respect to $$\mu$$:

$$
f_X(x) = \frac{d\mathbb{P}_X}{d\mu}(x)=\lim_{h\to 0}\frac{\mathbb{P}_X([x,x+h])}{\mu([x,x+h])}
$$

where $$\mathbb{P}_X$$ is absolutely continuous with respect to $$\mu$$.

### Expectation 

The **expectation** of a random variable $$X$$ is the Lebesgue integral of $$X$$ with respect to the probability measure $$\mathbb{P}$$. Formally:

$$
\mathbb{E}[X]=\int_{\Omega}X(\omega) d\mathbb{P}(\omega)
$$

We can also define expectation in terms of the pushforward measure $$\mathbb{P}_X$$ (the distribution of $$X$$):

$$
\mathbb{E}[X]=\int_{\mathbb{R}}x d\mathbb{P}_X(x)
$$

### Variance

The **variance** of a random variable $$X$$ measures the spread of $$X$$ around its mean. It is defined as the expectation of the squared deviation of $$X$$ from its mean:

$$
\text{Var}[X]=\int_{\Omega}(X(\omega)-\mathbb{E}[X(\omega)])^2 d\mathbb{P}(\omega)=\mathbb{E}[(X-\mathbb{E}[X])^2]
$$

With the pushforward measure $$\mathbb{P}_X$$, we can also defined variance as

$$
\text{Var}[X]=\int_{\mathbb{R}}(x-\mathbb{E}[X])^2 d\mathbb{P}_X(x)
$$