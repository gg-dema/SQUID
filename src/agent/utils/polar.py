import torch

def euclidean_to_polar(X):
    X_polar = X.clone()
    # r
    X_polar[..., 0] = torch.sqrt(X[..., 0]**2 + X[..., 1]**2)

    # theta
    X_polar[..., 1] = torch.arctan2(X[..., 0], X[..., 1])

    # ignore the rest of the system: remain std

    return X_polar

def polar_to_euclidean(X):
    X_euc = X.clone()
    X_euc[0] = X[..., 0] * torch.cos(X[..., 1])
    X_euc[1] = X[..., 0] * torch.sin(X[..., 1])
    return X_euc

def polar_to_euclidean_velocity(X, X_dot):
    """implement 
            dx_dt = dr_dt * np.cos(theta) - r * np.sin(theta) * dtheta_dt
            dy_dt = dr_dt * np.sin(theta) + r * np.cos(theta) * dtheta_dt
            + standard dyn equals between 2 systems
     """""
    X_dot_euc = torch.clone(X_dot)
    X_dot_euc[..., 0] = X_dot[..., 0] * torch.cos(X[..., 1]) - X[..., 0] * torch.sin(X[..., 1])
    X_dot_euc[..., 1] = X_dot[..., 0] * torch.sin(X[..., 1]) + X[..., 0] * torch.cos(X[..., 0])
    return X_dot_euc
