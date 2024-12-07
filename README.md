# Efficient 2:4 Sparse Pre-training Examples

This repository contains code of the following papers. For the latest version of our toolkit, please install from [https://github.com/huyz2023/2by4-pretrain](https://github.com/huyz2023/2by4-pretrain).

**Accelerating Transformer Pre-training with 2:4 Sparsity** [[arXiv]](https://arxiv.org/abs/2404.01847) [[OpenReview]](https://openreview.net/forum?id=kTaX87Zn6M) [[PDF]](https://proceedings.mlr.press/v235/hu24r.html)

Yuezhou Hu, Kang Zhao, Weiyu Huang, Jianfei Chen, Jun Zhu

International Conference on Machine Learning (ICML), 2024

**S-STE: Continuous Pruning Function for Efficient 2:4 Sparse Pre-training** [[arXiv]](https://arxiv.org/abs/2409.09099) [[OpenReview]](https://openreview.net/forum?id=8abNCVJs2j)

Yuezhou Hu, Jun Zhu, Jianfei Chen

Neural Information Processing Systems (NeurIPS), 2024

## Training

The different folders include different methods:

- `original`: Baselines we use in our papers (same baselines for both papers).
- `v1`: Transposable SR-STE + dense fine-tuning (from paper "Accelerating Transformer Pre-training with 2:4 Sparsity")
- `v2`: S-STE + FP8 (simulation only, from paper "S-STE: Continuous Pruning Function for Efficient 2:4 Sparse Pre-training")

By default, minimum-variance unbiased estimator (MVUE) is applied to backward pass to calculate `linear.weight.grad`.

### nanoGPT

We recommend you to run nanoGPT scripts since it applies the least modification to the original code and is easy to read. Our code is a copy from the original https://github.com/karpathy/nanoGPT.

**Common instructions:**

To replicate pre-training, enter `*/nanoGPT` folder and run:

```
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

For evaluation (GLUE and SQuAD), enter `*/nanoGPT/finetune_*` folder and run:

```
sh run.sh
```

We only provide scripts to replicate GPT-2 124M. To replicate other sizes in paper, follow the instructions below.

1. Model arguments: Modify`(n_layer, n_head, n_embd)` in `*/nanoGPT/train.py`. Specifically, `(12, 12, 768)` for 124M, `(24, 16, 1024)` for GPT-2 350M, `(36, 20, 1280)` for GPT-2 774M, and `(48, 25, 1600)` for GPT-2 1558M.
2. (`v1` only) Masked decay factor: Modify masked decay factor `alpha` in `v1/nanoGPT/train.py`.  For GPT-2 124M and 774M, this should be `6e-5`, for GPT-2 350M and GPT-2 1558M, this should be `2e-4`.
3. Evaluation script: Modify `hf_model` in `*/nanoGPT/finetune_*/run.sh` to match the model size (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`).

**Hyperparameters for pre-training and evaluation:**

| pre-training          | value  |
| --------------------- | ------ |
| learning rate         | 1.5e-4 |
| minimum learning rate | 1e-5   |
| batch size            | 512    |
| sequence length       | 1024   |
| max iters             | 300k   |
| warmup                | 3k     |

| evaluation    | value |
| ------------- | ----- |
| learning rate | 5e-6  |
| warmup        | 0.1   |

**What changes are made from the original nanoGPT repository?**

1. Change hyperparameters.

2. Use float16 and GradScaler for training stability.

3. Replace `nn.Linear` from FFN block with `FP8SparseLinear` or `SparseLinearTranspose`. The relevant code is in `sparse_ops.py`.

4. (`v1` only) Masked decay and dense fine-tuning:

   ```
    for micro_step in range(gradient_accumulation_steps):
    ...
       # clip the gradient
       if grad_clip != 0.0:
           scaler.unscale_(optimizer)
           torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
       #################### content added ####################
       with torch.no_grad():
           for p in model.parameters():
               if hasattr(p, 'mask') and p.mode == 'sparse':
                   p.grad = p.grad.float()
                   masked_add_(p.grad.data, p.data, p.mask, alpha=alpha)
                   p.cnt = 0
       if iter_num == 250000:
           for p in model.parameters():
               if hasattr(p, 'mask') and p.mode == 'sparse':
                   p.mode = 'dense'
       #################### content added ####################
       # step the optimizer and scaler if training in fp16
       scaler.step(optimizer)
       scaler.update()
       # flush the gradients as soon as we can, no need for this memory anymore
       optimizer.zero_grad(set_to_none=True)
   ...
   ```

## Citation

If you like our study, please cite:

```
@inproceedings{
  hu2024accelerating,
  title={Accelerating Transformer Pre-training with 2:4 Sparsity},
  author={Yuezhou Hu and Kang Zhao and Weiyu Huang and Jianfei Chen and Jun Zhu},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=kTaX87Zn6M}
}
@inproceedings{
  hu2024sste,
  title={S-{STE}: Continuous Pruning Function for Efficient 2:4 Sparse Pre-training},
  author={Yuezhou Hu and Jun Zhu and Jianfei Chen},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=8abNCVJs2j}
}
```
