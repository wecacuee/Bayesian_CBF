"""
Home for functions/classes that haven't find a home of their own
"""
import torch

def t_vstack(xs):
    torch.cat(xs, dim=-2)


