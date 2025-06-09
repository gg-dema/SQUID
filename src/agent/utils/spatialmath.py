import numpy as np
import matplotlib
from scipy.linalg import expm, logm


def skew_symmetric(v: np.array):
    """Convert a 3D vector into a skew-symmetric matrix."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def vee_map(S: np.array):
    """Convert a skew-symmetric matrix into a 3D vector."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def log_so3(R: np.array, R0=np.eye(3)):
    """
    Logarithmic map of SO(3), relative to a base rotation R0.
    Computes the element in the tangent space (Lie algebra so(3)) corresponding to R.

    :param R: A 3x3 rotation matrix.
    :param R0: The base rotation matrix (default is the identity matrix).
    :return: A 3D vector in the tangent space.
    """
    # Compute relative rotation R_rel = R0.T @ R
    R_rel = np.dot(R0.T, R)

    # Matrix logarithm of R_rel (this will be a skew-symmetric matrix)
    S = logm(R_rel)

    # Extract the 3D vector from the skew-symmetric matrix
    omega = vee_map(S)

    return omega

def exp_so3(omega: np.array, R0 = np.eye(3)):
    """
    Exponential map of SO(3), relative to a base rotation R0.
    Maps an element in the tangent space (Lie algebra so(3)) back to the group SO(3).

    :param omega: A 3D vector in the tangent space.
    :param R0: The base rotation matrix (default is the identity matrix).
    :return: A 3x3 rotation matrix.
    """
    # Create skew-symmetric matrix from omega
    S = skew_symmetric(omega)

    # Matrix exponential to map back to SO(3)
    R_rel = expm(S)

    # Compute the full rotation R = R0 @ exp(S)
    R = np.dot(R0, R_rel)

    return R
