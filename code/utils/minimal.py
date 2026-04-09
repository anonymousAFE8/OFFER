import torch.nn as nn


def normal_initialization(module, initial_range=0.02):
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initial_range)
        if module.padding_idx is not None:
            nn.init.constant_(module.weight.data[module.padding_idx], 0.0)
    elif isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=initial_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
