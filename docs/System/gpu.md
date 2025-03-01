---
layout: default
title: GPU
parent: System
---
# GPU

## Basic Concept

**Threads**. A thread is the smallest unit of execution in a GPU. Each thread runs the same program (kernel) but operates on different data. Threads execute SIMD (Single Instruction, Multiple Data) style: they run the same instruction but on different pieces of data.

**Warps**. A warp is a group of 32 threads (on NVIDIA GPUs) that execute in lockstep (meaning they execute the same instruction at the same time). If threads in a warp take different execution paths (due to conditionals like if statements), warp divergence occurs, slowing execution.

**Thread Blocks**. A thread block is a collection of multiple warps that can communicate through shared memory and synchronize using `__syncthreads()`. The number of threads per block is typically a multiple of 32 (since warps execute together), commonly 128, 256, or 512 threads per block.

**Grid**. A grid consists of multiple thread blocks. Each block operates independently, but all blocks together process the full dataset. Blocks in the grid cannot communicate directly but can use global memory to share data.

**Streaming Multiprocessor (SM)**. A Streaming Multiprocessor (SM) is a processing unit in a GPU where warps are scheduled and executed. Each SM has multiple CUDA cores, responsible for executing instructions. Each SM processes multiple warps simultaneously. Thread blocks are assigned to SMs, and their threads run within the CUDA cores of that SM.

### Hierarchy of Execution

```
Grid
 ├── Block 0
 │    ├── Warp 0 (32 threads)
 │    ├── Warp 1 (32 threads)
 │    ├── ...
 │    ├── Warp N
 ├── Block 1
 │    ├── Warp 0 (32 threads)
 │    ├── Warp 1 (32 threads)
 │    ├── ...
 │    ├── Warp N
 ├── ...
 ├── Block M

```

### Memory Types in GPU
- Global Memory: Accessible by all threads but slow (high latency).
- Shared Memory: Fast, block-wide memory for communication between threads within a block.
- Registers: Fast, thread-local memory (each thread has its own registers).
- Constant Memory: Read-only memory shared across all threads, optimized for frequently used values.

### Example: Kernel Execution
A CUDA kernel launch looks like:

```cpp
kernel<<<numBlocks, threadsPerBlock>>>(args);
```

where:
- `numBlocks` defines the number of thread blocks in the grid.
- `threadsPerBlock` defines the number of threads in each block.
For example:

```cpp
kernel<<<16, 256>>>(args);
```

This means:
- 16 blocks in the grid.
- 256 threads per block.
- Total threads: 16 × 256 = 4096.

### Table of Summary

| **Concept**         | **Definition** | **Relation to SM** |
|---------------------|---------------|--------------------|
| **Grid**           | Collection of all **thread blocks** for a kernel launch. | Distributed across multiple SMs. |
| **Thread Block**    | A group of threads that execute on the **same SM** and share **shared memory**. | Each **SM** runs one or more thread blocks. |
| **Warp**           | A **group of 32 threads** that execute together. | An SM schedules and executes warps. |
| **Thread**         | Smallest unit of execution. | Runs inside a warp on an SM's **CUDA cores**. |

### Example: A100

The A100 has 108 SMs, each with:

- 64 CUDA cores (for general-purpose computation)
- 4 Tensor cores (for matrix multiplications)
- Warp schedulers and execution units
- 164 KB of shared memory

Each SM schedules and executes multiple warps in parallel. When a CUDA kernel is launched, it follows this hierarchical execution model:
1. A single grid is created per kernel launch.
2. The grid consists of multiple thread blocks.
3. Each thread block is assigned to an SM.
4. The SM divides the thread block into warps (32 threads per warp).
5. The warp scheduler schedules and executes warps across CUDA cores.

A100 has a hierarchical memory system that impacts performance:

- Global Memory (HBM2e, 40 GB, slowest)
- L2 Cache (40 MB, shared across SMs)
- Shared Memory (164 KB per SM, fast)
- Registers (per-thread storage, fastest)

Threads within a block can share data using shared memory, reducing access to slow global memory. There are also a couple of performance considerations

1. Occupancy Optimization 
- The more active warps per SM, the better the GPU hides memory latency.
- A100 supports up to 64 warps per SM (64 × 32 = 2,048 active threads per SM).
2. Memory Coalescing
- Access global memory efficiently by using aligned 128-byte transactions.
- Use shared memory for frequent data access.
3. Tensor Cores for AI Workloads
- Tensor cores accelerate matrix multiplications (e.g., FP16, TF32, INT8 operations).
They enable faster deep learning training & inference.


