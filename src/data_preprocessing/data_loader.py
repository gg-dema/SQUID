from datasets.dataset_keys import (
    LASA, LAIR,
    optitrack, interpolation, kuka,
    joint_space, multi_attractors,
    cycle, eval_dataset
)

from spatialmath import SO3, UnitQuaternion
import os
import pickle
import numpy as np
import scipy.io as sio


def load_demonstrations(dataset_name, selected_primitives_ids, dim_manifold):
    """
    Loads demonstrations
    """
    # Get names of primitives in dataset
    dataset_primitives_names = get_dataset_primitives_names(dataset_name)

    # Get names of selected primitives for training
    primitives_names, primitives_save_name = select_primitives(dataset_primitives_names, selected_primitives_ids)

    # Get number of selected primitives
    n_primitives = len(primitives_names)

    # Get loading path
    dataset_path = 'datasets/' + dataset_name + '/'

    # Get data loader
    data_loader = get_data_loader(dataset_name, dim_manifold)

    # Load
    demonstrations, demonstrations_primitive_id, delta_t_eval = data_loader(dataset_path, primitives_names)

    # Out dictionary
    loaded_info = {'demonstrations raw': demonstrations,
                   'demonstrations primitive id': demonstrations_primitive_id,
                   'n primitives': n_primitives,
                   'delta t eval': delta_t_eval}
    return loaded_info


def get_dataset_primitives_names(dataset_name):
    """
    Chooses primitives keys
    """
    if dataset_name == 'LASA':
        dataset_primitives_names = LASA
    elif dataset_name == 'LAIR':
        dataset_primitives_names = LAIR
    elif dataset_name == 'optitrack':
        dataset_primitives_names = optitrack
    elif dataset_name == 'kuka':
        dataset_primitives_names = kuka
    elif dataset_name == 'interpolation':
        dataset_primitives_names = interpolation
    elif dataset_name == 'joint_space':
        dataset_primitives_names = joint_space
    elif dataset_name == 'multi_attractors':
        dataset_primitives_names = multi_attractors
    elif dataset_name == 'eval_dataset':
        dataset_primitives_names = eval_dataset
    else:
        raise NameError('Dataset %s does not exist' % dataset_name)

    return dataset_primitives_names


def select_primitives(dataset, selected_primitives_ids):
    """
    Gets selected primitives
    """
    selected_primitives_names = []
    selected_primitives_save_name = ''
    selected_primitives_ids = list(map(int, selected_primitives_ids.split(',')))  # map from string to list
    for id in selected_primitives_ids:
        selected_primitives_names.append(dataset[id])
        selected_primitives_save_name += str(id) + '_'

    return selected_primitives_names, selected_primitives_save_name[:-1]


def get_data_loader(dataset_name, dim_manifold):
    """
    Chooses data loader depending on the data type
    """
    dataset_in_numpy_format = [
        'optitrack',
        'LAIR',
        'interpolation',
        'eval_dataset',
        'multi_attractors',
        'simple',
        'cycle',
    ]
    if dataset_name == 'LASA':
        data_loader = load_LASA
    elif dataset_name in dataset_in_numpy_format:
        data_loader = load_numpy_file
    elif dataset_name == 'joint_space':
        data_loader = load_from_dict
    elif dataset_name == 'kuka' and dim_manifold == 3:
        data_loader = load_R3
    elif dataset_name == 'kuka' and dim_manifold == 7:
        data_loader = load_R3S3
    else:
        raise NameError('Dataset %s does not exist' % dataset_name)

    return data_loader


def load_LASA(dataset_dir, demonstrations_names):
    """
    Load LASA matlab models
    """
    s_x, s_y, demos, primitive_id, dt = [], [], [], [], []
    for i in range(len(demonstrations_names)):
        mat_file = sio.loadmat(dataset_dir + demonstrations_names[i])
        data = mat_file['demos']

        for j in range(data.shape[1]):  # iterate through demonstrations
            s_x = data[0, j]['pos'][0, 0][0]
            s_y = data[0, j]['pos'][0, 0][1]
            s = [s_x, s_y]
            demos.append(s)
            dt.append(data[0, j]['dt'][0, 0][0, 0])
            primitive_id.append(i)

    return demos, primitive_id, dt


def load_numpy_file(dataset_dir, demonstrations_names):
    """
    Loads demonstrations in numpy files
    """
    demos, primitive_id = [], []
    for i in range(len(demonstrations_names)):
        demos_primitive = os.listdir(dataset_dir + demonstrations_names[i])

        for demo_primitive in demos_primitive:
            data = np.load(dataset_dir + demonstrations_names[i] + '/' + demo_primitive)
            if data.shape[0] == 1:
                # if extra dimension in demo, remove
                data = data[0]
            demos.append(data.T)
            primitive_id.append(i)

    dt = 1
    return demos, primitive_id, dt


def load_from_dict(dataset_dir, demonstrations_names):
    """
    Loads demonstrations in dictionaries
    """
    demos, primitive_id, dt = [], [], []

    # Iterate in each primitive (multi model learning)
    for i in range(len(demonstrations_names)):
        demos_primitive = os.listdir(dataset_dir + demonstrations_names[i])

        # Iterate over each demo in primitive
        for demo_primitive in demos_primitive:
            filename = dataset_dir + demonstrations_names[i] + '/' + demo_primitive
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            demos.append(data['q'].T)
            dt.append(data['delta_t'])
            primitive_id.append(i)

    return demos, primitive_id, dt

def load_R3S3(dataset_dir, demonstrations_names):

    demos, primitive_id, dt = [], [], []
    for i in range(len(demonstrations_names)):

        demos_primitive = os.listdir(dataset_dir + demonstrations_names[i])
        for demo_primitive in demos_primitive:
            filename = dataset_dir + demonstrations_names[i] + '/' + demo_primitive

            with open(filename, 'rb') as file:
                data = pickle.load(file)

            prev_quat = None
            quats = []
            for numpy_rot_mat in data['x_rot']:
                quat = UnitQuaternion(SO3(numpy_rot_mat)).A

                # check for flip in quat trajectory (singular representation):
                if prev_quat is None:  prev_quat = quat
                dist_quats = np.linalg.norm(quat - prev_quat)
                if dist_quats > 0.5:
                    quat *= -1

                # store intermediate quat
                quats.append(quat)
                prev_quat = quat

            # 1 demo = pos and orientation --> concat
            positions = np.array(data['x_pos'])
            demo = np.concatenate([positions, quats], axis=1)

            # store demos
            demos.append(demo.T)
            dt.append(data['delta_t'])
            primitive_id.append(i)

    return demos, primitive_id, dt


def load_R3(dataset_dir, demonstrations_names):
    """
    Loads demonstrations in 3D
    """
    demos, primitive_id, dt = [], [], []

    # Iterate in each primitive (multi model learning)
    for i in range(len(demonstrations_names)):
        demos_primitive = os.listdir(dataset_dir + demonstrations_names[i])

        # Iterate over each demo in primitive
        for demo_primitive in demos_primitive:
            filename = dataset_dir + demonstrations_names[i] + '/' + demo_primitive
            with open(filename, 'rb') as file:
                data = pickle.load(file)

            # this is pre-processing. Should be done in the preprocessing file.
            # moving the trajectory shifting there
            # Set zero as goal
            # Not to much sense for a double attractor system
            # positions = np.array(data['x_pos']) - np.array(data['x_pos'])[-1]

            # Append demo to demo list
            demos.append(np.array(data['x_pos']).T)
            dt.append(data['delta_t'])
            primitive_id.append(i)

    return demos, primitive_id, dt
