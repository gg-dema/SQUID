import numpy as np
from spatialmath import SO3, UnitQuaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')
from scipy.linalg import expm, logm
import pickle

from osmp.spatialmath import log_so3, exp_so3


if __name__ == "__main__":
    # Load data
    data_name = 'ee_state__one_loop_orientation_1.pk'
    data = np.load(data_name, allow_pickle=True)
    orientation_so3 = data['x_rot']

    # Compute origin tangent plane
    orientation_quat = [SO3(orientation).UnitQuaternion().A for orientation in data['x_rot']]
    mean_quaternion = np.array(orientation_quat).mean(axis=0)
    mean_quaternion = mean_quaternion / np.linalg.norm(mean_quaternion)
    orientation_origin = UnitQuaternion(mean_quaternion).SO3().A

    # Compute log map
    log_so3 = np.array([log_so3(rotation, orientation_origin) for rotation in orientation_so3])
    data['x_log_rot'] = list(log_so3)
    data['log_origin'] = orientation_origin

    # Compute velocity in tangent plane
    delta_t = data['delta_t']
    log_so3_dot = [(log_so3[i + 1] - log_so3[i]) / delta_t[i + 1] for i in range(len(log_so3) - 1)]
    log_so3_dot.insert(0, log_so3_dot[0])
    data['x_log_rot_dot'] = log_so3_dot

    # Save values
    with open('log_' + data_name, 'wb') as file:
        pickle.dump(data, file)

    # Compute exp map for sanity check
    exp_q = np.array([exp_so3(rotation_log, orientation_origin) for rotation_log in list(log_so3)])
    print('Log exp error:', np.mean(np.array(orientation_so3) - exp_q))

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(log_so3[:, 0], log_so3[:, 1], log_so3[:, 2], c='b', marker='o')

    # Labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Trajectory in Tangent Plane')

    # Show plot
    plt.show()
