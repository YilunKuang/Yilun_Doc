---
layout: default
title: Sequence Models
parent: Machine Learning
---

# Sequence Models

## Hyena

Convolution:

```python
def conv(u, k):
    u_f = torch.fft.fft(u)
    k_f = torch.fft.fft(k)
    y_f = u_f * k_f
    return torch.fft.ifft(y_f)
```
