import math

import torch


def exp_map(x, v):
    """ exponential map on the s^n sphere --> from tanget space to manifold"""
    norm_v = torch.linalg.norm(v)
    if norm_v < 1e-10:
        return x  # No movement
    return torch.cos(norm_v) * x + torch.sin(norm_v) * (v / norm_v)


def torus_exp_map(y_t, v, latent_space_dim=300):
    """
    Torus exponential map: move along v from y_t and wrap.

    Args:
        y_t: Tensor [B, D] - base points on the torus
        v: Tensor [B, D]   - tangent vectors

    Returns:
        new_points: Tensor [B, D] - points moved along tangent and wrapped to T^D
    """

    return (y_t + v + torch.pi) % (2 * torch.pi) - torch.pi
    #return torch.norm(log_vect, dim=-1)/math.sqrt(latent_space_dim)

def torus_log_map(x, g):
    """
    Compute log map vector from g to x on torus: wrap(x - g) ∈ [-π, π]
    x: [B, D], g: [n_goals, D] or [B, D]
    returns: [B, n_goals, D]
    """
    pi = torch.tensor(math.pi, device=x.device)
    delta = (x[:, None, :] - g[None, :, :] + pi) % (2 * pi) - pi
    return delta
def dist_torus_log_map(y_1, y_2, normed_dist=True):
    """ used also as metrics for the distance between element in the tangent space"""
    diff = (y_1 - y_2.unsqueeze(-1) + torch.pi) % (2 * torch.pi) - torch.pi      # [B, D, N]
    dist = torch.norm(diff, dim=1)                                               # [B, N]
    return dist
