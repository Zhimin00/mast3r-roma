#!/bin/bash
#SBATCH --job-name=trainwarp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4             # 每节点 8 个进程，对应 8 张 GPU
#SBATCH --cpus-per-task=8               # 每个 GPU 分配的 CPU 数量
#SBATCH --gres=gpu:4                    # 每节点使用 4 张 GPU
#SBATCH --time=64:00:00
#SBATCH --partition=a100      
#SBATCH --output=/home/cpeng26/scratchrchella4/logs/%x_%j.out
#SBATCH --error=/home/cpeng26/scratchrchella4/logs/%x_%j.err

source ~/.bashrc
conda activate mast3r

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500
NODE_RANK=$SLURM_NODEID
NNODES=$SLURM_JOB_NUM_NODES

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=4 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_warp.py \
    --gpu_batch_size=12