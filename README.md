# Accelerating Transformer Pre-training with 2:4 Sparsity
This repository contains code to replicate our research described in "Accelerating Transformer Pre-training with 2:4 Sparsity".

arxiv: [https://arxiv.org/abs/2404.01847](https://arxiv.org/abs/2404.01847)

For the latest version of our toolkit, please refer to [https://github.com/huyz2023/2by4-pretrain](https://github.com/huyz2023/2by4-pretrain)

## Training

### Cramming

[https://github.com/JonasGeiping/cramming](https://github.com/JonasGeiping/cramming)

Note:

1. We use the previous version of Cramming, which is deprecated now. The dataset is no longer available to reproduce our baseline. We will later fix this.
2. Setting up the environment is tricky. We recommend using `flash-attn==1.0.8` if compilation fails.

Usage:

1. Follow the instructions in `cramming/setup_env.sh` to set up the environment, and make sure nvcc version and cuda version of pytorch match.
2. Prepare data.
3. For pre-training, follow each `start.sh` scripts in `cramming/*` folders.
4. For fine-tuning, `cramming/*-eval` folders are prepared for Bi-Mask, STEP and SR-STE, and for Dense and Half we use the same directory as pre-training. Move the `outputs` folder to one of these working directories, and follow `start.sh` in each of them.

### DeiT

https://github.com/facebookresearch/deit

Usage:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_224 --batch-size 256 --data-path  /root/autodl-tmp/imagenet --output_dir /root/autodl-tmp/outputs
```

### nanoGPT

https://github.com/facebookresearch/deit

Usage:

```
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py --compile=False
```

### Acceleration results

Follow ` accelerate/run_ffn.sh ` and `accelerate/run_block.sh`.

## Citation

If you use this codebase or find our work interesting, please consider citing:

```
@misc{hu2024accelerating,
      title={Accelerating Transformer Pre-Training with 2:4 Sparsity}, 
      author={Yuezhou Hu and Kang Zhao and Weiyu Huang and Jianfei Chen and Jun Zhu},
      year={2024},
      eprint={2404.01847},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

