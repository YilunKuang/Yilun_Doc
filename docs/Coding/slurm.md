---
layout: default

title: Slurm

parent: Coding
---
# Slurm



## Bash Function

helpful functions to request compute resources

```bash
# Function to request resources with srun
gpu() {
    if [ $# -lt 2 ]; then
        echo "Usage: gpu <time_in_hours> <memory_in_gb>"
        return 1
    fi

    local hours="$1"
    local memory_gb="$2"

    # Convert hours to minutes
    local minutes=$((hours * 60))

    # Convert memory from GB to MB
    local memory_mb=$((memory_gb * 1024))

    srun -t "${minutes}:00" --mem="${memory_mb}" --gres=gpu:1 --pty /bin/bash

}

# Function to request resources with srun
gpu_long() {
    if [ $# -lt 5 ]; then
        echo "Usage: gpu_long <time_in_hours> <memory_in_gb> <num_nodes> <num_gpus> <gpu_type>"
        return 1
    fi

    local hours="$1"
    local memory_gb="$2"
    local num_nodes="$3"
    local num_gpus="$4"
    local gpu_type="$5"

    # Convert hours to minutes
    local minutes=$((hours * 60))

    # Convert memory from GB to MB
    local memory_mb=$((memory_gb * 1024))

    # Request the specified GPUs and GPU type on the specified number of nodes
    srun -t "${minutes}:00" --mem="${memory_mb}" --nodes="${num_nodes}" --ntasks-per-node="${num_gpus}" --gres="gpu:${gpu_type}:${num_gpus}" --pty /bin/bash
}

# Function to request CPU resources with srun
cpu() {
    if [ $# -lt 2 ]; then
        echo "Usage: cpu <time_in_hours> <memory_in_gb>"
        return 1
    fi

    local hours="$1"
    local memory_gb="$2"

    # Convert hours to minutes
    local minutes=$((hours * 60))

    # Convert memory from GB to MB
    local memory_mb=$((memory_gb * 1024))

    srun --nodes=1 --tasks-per-node=1 --cpus-per-task=1 -t "${minutes}:00" --mem="${memory_mb}" --pty /bin/bash

}

```