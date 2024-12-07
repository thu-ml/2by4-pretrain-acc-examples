import torch

L = torch.load("out/ckpt.pt")
print(L['model_args'])