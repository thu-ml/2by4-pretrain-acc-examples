#!/bin/bash

USE_CUSPARSELT=1
CUSPARSELT_ROOT=/home/bingxing2/home/scx6117/soft/para/libcusparse_lt-linux-sbsa-0.5.2.1-archive
export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_PREFIX_PATH=/home/bingxing2/home/scx6117/.conda/envs/cuda118py311
module load compilers/cuda/11.8 cudnn/8.6.0.163_cuda11.x anaconda/2021.11 compilers/gcc/9.3.0
source activate huyz

### torchrun --standalone --nproc_per_node=1 train.py config/train_gpt2.py --compile=False --wandb_log=True  > "/home/bingxing2/home/scx6117/huyz/nanoGPT/log.txt" 2>&1



### 启用 IB 通信
export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

### 获取每个节点的 hostname

for i in `scontrol show hostnames`
do
let k=k+1
host[$k]=$i
echo ${host[$k]}
done

#torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


#### 主节点运行
torchrun --nproc_per_node=4 --nnodes=8 --node_rank=0 --master_addr="${host[1]}" --master_port=1234 train.py config/train_gpt2.py --compile=False --wandb_log=True --init_from=resume >> train_rank0_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 --gpus=4 -w ${host[2]} torchrun --nproc_per_node=4 --nnodes=8 --node_rank=1 --master_addr="${host[1]}" --master_port=1234 train.py config/train_gpt2.py --compile=False --wandb_log=True --init_from=resume >> train_rank1_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 --gpus=4 -w ${host[3]} torchrun --nproc_per_node=4 --nnodes=8 --node_rank=2 --master_addr="${host[1]}" --master_port=1234 train.py config/train_gpt2.py --compile=False --wandb_log=True --init_from=resume >> train_rank2_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 --gpus=4 -w ${host[4]} torchrun --nproc_per_node=4 --nnodes=8 --node_rank=3 --master_addr="${host[1]}" --master_port=1234 train.py config/train_gpt2.py --compile=False --wandb_log=True --init_from=resume >> train_rank3_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 --gpus=4 -w ${host[5]} torchrun --nproc_per_node=4 --nnodes=8 --node_rank=4 --master_addr="${host[1]}" --master_port=1234 train.py config/train_gpt2.py --compile=False --wandb_log=True --init_from=resume >> train_rank4_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 --gpus=4 -w ${host[6]} torchrun --nproc_per_node=4 --nnodes=8 --node_rank=5 --master_addr="${host[1]}" --master_port=1234 train.py config/train_gpt2.py --compile=False --wandb_log=True --init_from=resume >> train_rank5_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 --gpus=4 -w ${host[7]} torchrun --nproc_per_node=4 --nnodes=8 --node_rank=6 --master_addr="${host[1]}" --master_port=1234 train.py config/train_gpt2.py --compile=False --wandb_log=True --init_from=resume >> train_rank6_${SLURM_JOB_ID}.log 2>&1 &
srun -N 1 --gres=gpu:4 --gpus=4 -w ${host[8]} torchrun --nproc_per_node=4 --nnodes=8 --node_rank=7 --master_addr="${host[1]}" --master_port=1234 train.py config/train_gpt2.py --compile=False --wandb_log=True --init_from=resume >> train_rank7_${SLURM_JOB_ID}.log 2>&1 &



wait


#sbatch -N 8 -p vip_gpu_scx6117 --gres=gpu:4 --gpus=32 --qos=gpugpu ./run.sh
#sbatch --gpus=4 ./run.sh
#squeue
#sinfo
#scancel
#squeue -i 1