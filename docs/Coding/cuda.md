---
layout: default

title: CUDA

parent: Coding
---
# CUDA

## Resource
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Book: Professional CUDA C Programming](https://www.cs.utexas.edu/~rossbach/cs380p/papers/cuda-programming.pdf)
- [GitHub: CUDA_Freshman](https://github.com/Tony-Tan/CUDA_Freshman)
- [Class: Stanford CS149 Parallel Computing](https://gfxcourses.stanford.edu/cs149/fall23/)
    - [Assignment 4: NanoGPT149](https://github.com/stanford-cs149/cs149gpt/tree/main)
- [Book: An Introduction to Parallel Programming](https://www.cs.usfca.edu/%7Epeter/ipp/)
- [YouTube: CoffeeBeforeArch - CUDA Crash Course](https://www.youtube.com/playlist?list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU)
- [GitHub: Andrej Karpathy - llm.c](https://github.com/karpathy/llm.c/tree/master)

## Learning

### YouTube: CoffeeBeforeArch - CUDA Crash Course

- [x] 1 CUDA Crash Course: Vector Addition
    - [reference code](https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/01_vector_addition/baseline/vectorAdd.cu)
        - Single Instruction Multiple Threads (SIMT) Model
            - threads -> warps -> thread blocks -> grids
        - allocate memory on device 
            - `cudaMalloc(&d_a, bytes); cudaMalloc(&d_b, bytes); cudaMalloc(&d_c, bytes);`
        - copy data from host to device
            - `cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);`
            - `cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);`
        - set thread sizes; set grid sizes
            - `int NUM_THREADS = 256;`
            - `int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);`
        - launch kernel on default stream w/o shmem
            - `vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);`
        - copy data from device to host
            - `cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);`
        - CUDA kernel
            - `__global__` declaration for cuda kernel
            - define global thread ID (tid) 
                - `int tid = (blockIdx.x * blockDim.x) + threadIdx.x;`
        - free memory
            `cudaFree(d_a);`
- [x] 2 CUDA Crash Course: Unified Memory Vector Add
    - [reference code: unified memory](https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/01_vector_addition/unified_memory/vectorAdd_um_baseline.cu)
        - unified memory
            - `cudaMallocManaged(&a, bytes);`
        - sync after CUDA Kernel
            - CUDA Kernel launching is asynchronous so we need to sync
            - for `cudaMalloc`, `cudaMemcpy` serves as our synchronization barrier, but for `cudaMallocManaged` we have to explicitly set a synchronization point.
            - synchronization
                - `cudaDeviceSynchronize();`
    - [reference code: prefetching](https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/01_vector_addition/unified_memory/vectorAdd_um_prefetch.cu)
        - before launching cuda kernel, prefetch
            - prefetch from host to device
                - `cudaMemPrefetchAsync(a, bytes, id); cudaMemPrefetchAsync(b, bytes, id);`
            - prefetch from device to host
                - `cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);`
- [ ] 3 CUDA Crash Course: Matrix Multiplication
    - TODO! 