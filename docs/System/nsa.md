---
layout: default
title: Native Sparse Attention
parent: System
---
# Native Sparse Attention


**Core Idea**
- NSA = Query-Aware Dynamic Sparse Attention

**Algorithm**
- NSA = Token Compression + Token Selection + Sliding Window

**Kernel Design**
- Grouping query blocks by heads instead of adjacent tokens as in flash attention

**Observations**
- NSA is compatible with MLA, i.e. GQA-based NSA is compatible with MLA
- In terms of KV size, NSA interpolates between full KV cache in full attention and a single accumulated KV in linear attention variants
