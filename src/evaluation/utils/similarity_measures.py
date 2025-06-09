import numpy as np
import similaritymeasures as sm
from scipy.spatial import cKDTree

def get_RMSE(sim_trajectories, demos, eval_indexes=None, verbose=True):
    RMSE = []
    for k in range(sim_trajectories.shape[1]):
        if verbose:
            print('Calculating RMSE; trajectory:', k + 1)
        if eval_indexes is not None:
            RMSE.append(
                np.sqrt(
                    np.mean(
                        (sim_trajectories[eval_indexes[k], k, :] - demos[eval_indexes[k], k, :]) ** 2)))
        else:
            RMSE.append(np.sqrt(np.mean((sim_trajectories[:, k, :] - demos[:, k, :]) ** 2)))
    return RMSE


def get_DTWD(sim_trajectories, demos, eval_indexes, verbose=True):
    DTWD = []
    for k in range(sim_trajectories.shape[1]):
        if verbose:
            print('Calculating dynamic time warping distance; trajectory:', k + 1)
        dtw, d = sm.dtw(sim_trajectories[eval_indexes[k], k, :], demos[eval_indexes[k], k, :])
        DTWD.append(dtw / len(eval_indexes[k]))

    return DTWD


def get_FD(sim_trajectories, demos, eval_indexes, verbose=True):
    FD = []
    for k in range(sim_trajectories.shape[1]):
        if verbose:
            print('Calculating Frechet distance; trajectory:', k + 1)
        FD.append(sm.frechet_dist(sim_trajectories[eval_indexes[k], k, :], demos[eval_indexes[k], k, :]))

    return FD

def get_chamfer_distance(attractors, reference_shape):
    attractors_tree = cKDTree(attractors)
    reference_tree = cKDTree(reference_shape)

    d1, _ = attractors_tree.query(reference_shape)
    d2, _ = reference_tree.query(attractors)

    chamfer = np.mean(d1**2) + np.mean(d2**2)
    return chamfer

def get_spurious_attractors_shape(attractors,
                                  reference_shape,
                                  distance_threshold):
    shape_tree = cKDTree(reference_shape)
    distance, _ = shape_tree.query(attractors, k=1)
    spurious_attractors_mask = distance < distance_threshold
    return np.sum(~spurious_attractors_mask)
