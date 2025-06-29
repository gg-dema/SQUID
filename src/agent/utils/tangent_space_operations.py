

import torch


def exp_map(x, v):
    """ exponential map on the s^n sphere --> from tanget space to manifold"""
    norm_v = torch.linalg.norm(v)
    if norm_v < 1e-10:
        return x  # No movement
    return torch.cos(norm_v) * x + torch.sin(norm_v) * (v / norm_v)
