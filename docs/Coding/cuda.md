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
- [x] 3 CUDA Crash Course: Matrix Multiplication
    - [reference code: matrix multiplication](https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/02_matrix_mul/baseline/mmul.cu)
        - Basic Flow
            - For square matrices multiplications $$A\times B =C$$ with $$A,B,C\in\mathbb{R}^{n\times n}$$, we would like to assign one thread for $$C[i][j]\space\forall i,j$$. Each thread would write the results of $$A[i,:]^\top B[:,j]$$ into $$C[i][j]$$.
        - 2D Indexing for Thread Blocks
            - Row = blockIdx.y * blockDim.y + threadIdx.y
            - Col = blockIdx.x * blockDim.x + threadIdx.x
        - Coalescing
            - 2D matrices are layered out as one array in memory. It's important to make sure that we're not bottlenecked by fragmented memory access.
        - Code
            - Matrix Size (1024 * 1024)
                - `int n = 1 << 10;` 
            - Threads Per Blocks (2D)
                - `int BLOCK_SIZE = 16;`
            - Blocks in Each Dimension
                - `int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);`
            - dim3 objects
                - `dim3 grid(GRID_SIZE, GRID_SIZE)`
                - `dim3 threads(BLOCK_SIZE, BLOCK_SIZE)`
                - This is 64*64*16*16 = 1024*1024 total number of threads
            - within the CUDA Kernel for matrix multiplication
                - `int row = blockIdx.y * blockDim.y + threadIdx.y;`
                - `int col = blockIdx.x * blockDim.x + threadIdx.x;`
                - basically, CUDA helps us reduce two for loop (loops over rows and columns) by parallelizing the computation of $$C[i][j]$$ across all threads. 
- [ ] 4 CUDA Crash Course: Cache Tiled Matrix Multiplication
    - [refernce code: tiled matrix multiplication](https://github.com/CoffeeBeforeArch/cuda_programming/blob/master/02_matrix_mul/tiled/mmul.cu)
    - TODO: Need to watch again and summarize here
 