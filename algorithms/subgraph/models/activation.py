import torch.nn as nn

activation_registry = dict(
    ELU=nn.ELU,
    ReLU=nn.ReLU,
    GELU=nn.GELU,
)
