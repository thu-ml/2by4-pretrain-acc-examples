import torch

from model import GPTConfig, GPT
from transformers import AutoModelForSequenceClassification


def load_hf_model(config):
    model_args = dict(n_layer=None, n_head=None, n_embd=None, block_size=None,
                      bias=None, vocab_size=None, dropout=None)
    device = 'cuda:0'
    checkpoint = torch.load('../out/ckpt.pt', map_location=device)
    checkpoint_model_args = checkpoint['model_args']

    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout']:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    raw_model = GPT(gptconf)
    state_dict = checkpoint['model']

    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    raw_model.load_state_dict(state_dict)

    model = AutoModelForSequenceClassification.from_config(
        config=config
    )

    model.transformer.wte = raw_model.transformer.wte
    model.transformer.wpe = raw_model.transformer.wpe
    model.transformer.drop = raw_model.transformer.drop

    for i in range(config.n_layer):
        model.transformer.h[i].ln_1 = raw_model.transformer.h[i].ln_1
        model.transformer.h[i].attn.c_attn = raw_model.transformer.h[i].attn.c_attn
        model.transformer.h[i].attn.c_proj = raw_model.transformer.h[i].attn.c_proj
        model.transformer.h[i].attn.attn_dropout = raw_model.transformer.h[i].attn.attn_dropout
        model.transformer.h[i].attn.resid_dropout = raw_model.transformer.h[i].attn.resid_dropout
        model.transformer.h[i].ln_2 = raw_model.transformer.h[i].ln_2
        model.transformer.h[i].mlp.c_fc = raw_model.transformer.h[i].mlp.c_fc
        model.transformer.h[i].mlp.c_proj = raw_model.transformer.h[i].mlp.c_proj
        model.transformer.h[i].mlp.dropout = raw_model.transformer.h[i].mlp.dropout

    model.transformer.ln_f = raw_model.transformer.ln_f

    return model
