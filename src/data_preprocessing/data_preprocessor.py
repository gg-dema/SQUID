import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
#from scipy.spatial.transform import Rotation as R
#from spatialmath import SO3, UnitQuaternion

from agent.utils.dynamical_system_operations import normalize_state
#from agent.utils.spatialmath import log_so3

from data_preprocessing.data_loader import load_demonstrations
# extra
from sklearn.cluster import KMeans
import os



class DataPreprocessor:
    def __init__(self, params, verbose=True):
        """
        Class for loading and preprocessing demonstrations
        """
        self.trajectories_resample_length = params.trajectories_resample_length
        self.state_increment = params.state_increment
        self.dim_workspace = params.workspace_dimensions
        self.dynamical_system_order = params.dynamical_system_order
        self.dim_state = self.dim_workspace * self.dynamical_system_order
        self.workspace_boundaries_type = params.workspace_boundaries_type
        self.workspace_boundaries = np.array(params.workspace_boundaries)
        self.eval_length = params.evaluation_samples_length
        self.dataset_name = params.dataset_name
        self.shaped_attractors = params.shaped_attractors_form
        self.selected_primitives_id = params.selected_primitives_ids
        self.spline_sample_type = params.spline_sample_type


        self.delta_t = 1  # this value is for learning, so it can be anything
        self.imitation_window_size = params.imitation_window_size  # window size used for imitation cost
        self.verbose = verbose

        # from here : MULTICONDOR
        self.n_attractors = params.n_attractors

        # from here: MULIT-PUMA:
        self.augment_data_orientation = False
        if self.augment_data_orientation:
            self.n_attractors *= 2

    def run(self):
        """
        Computes relevant features from the raw demonstrations
        """
        # Load demonstrations and associated data
        loaded_data = load_demonstrations(self.dataset_name, self.selected_primitives_id, self.dim_workspace)

        # Get features from demonstrations demonstrations
        #loaded_data, features_demos = self.get_features_demos(loaded_data)
        features_demos = self.get_features_demos(loaded_data)
        # Generate training data
        demonstrations_train = self.generate_training_data(loaded_data, features_demos)

        # Get min/max derivatives of training demonstrations
        limits_derivatives = self.get_limits_derivatives(demonstrations_train)

        # Create preprocess output dictionary
        preprocess_output = {'demonstrations train': demonstrations_train}
        preprocess_output.update(loaded_data)
        preprocess_output.update(features_demos)
        preprocess_output.update(limits_derivatives)

        return preprocess_output

    def augment_orientation(self, loaded_data):

        # loaded data = dict_keys(['demonstrations raw', 'demonstrations primitive id', 'n primitives', 'delta t eval'])
        #                     x  y  z |roll pitch yaw     ---> invert just the quaternion
        flip_mask = np.array([1, 1, 1, -1, -1, -1, -1])
        augmented_traj = [
            loaded_data['demonstrations raw'][i] * flip_mask[:, np.newaxis] for i in range(len(loaded_data['demonstrations raw']))
        ]
        augmented_data = {
            'demonstrations raw': loaded_data['demonstrations raw'] + augmented_traj,
            'demonstrations primitive id': loaded_data['demonstrations primitive id']*2,
            'n primitives': loaded_data['n primitives'],
            'delta t eval': loaded_data['delta t eval'] * 2,
        }
        return augmented_data
    '''
    REMOVED FOR TEST
    def map_orientation_to_tangent_space(self, loaded_data):
        demo_tangent_space = []

        for i in range(len(loaded_data['demonstrations raw'])):
            demo = loaded_data['demonstrations raw'][i]
            pos, quat = demo[:3, :], demo[3:, :]
            log_so3_demo = np.array(
                [log_so3(UnitQuaternion(quat[:, i]).SO3().A) for i in range(quat.shape[1]) ]
            ).T
            demo_tangent_space.append(np.concatenate((pos, log_so3_demo), axis=0))

        loaded_data['S3R3 demo raw'] = loaded_data['demonstrations raw']
        loaded_data['demonstrations raw'] = demo_tangent_space
        return loaded_data

    def map_orientation_to_euler_angle(self, loaded_data):
        demo_euler_angle = []
        for i in range(len(loaded_data['demonstrations raw'])):
            demo = loaded_data['demonstrations raw'][i]
            pos, quat = demo[:3, :], demo[3:, :]

            euler = R.from_quat(quat.T).as_euler('xyz', degrees=False).T
            euler = np.unwrap(euler, axis=1)
            demo_euler_angle.append(np.concatenate((pos, euler), axis=0))

        loaded_data['S3R3 demo raw'] = loaded_data['demonstrations raw']
        loaded_data['demonstrations raw'] = demo_euler_angle

        return loaded_data

    def fix_discontinuity_tg_space(self, loaded_data):
        broken_demo = [11, 2, 8, 15, 19]
        new_demo = []
        s3r3_demo_raw = []
        delta_t_eval = []
        prim_id = []
        for i in range(len(loaded_data['demonstrations raw'])):
            if i not in broken_demo:
                prim_id.append(loaded_data['demonstrations primitive id'][i])
                new_demo.append(loaded_data['demonstrations raw'][i])
                s3r3_demo_raw.append(loaded_data['S3R3 demo raw'][i])
                delta_t_eval.append(loaded_data['delta t eval'][i])

        loaded_data['S3R3 demo raw'] = s3r3_demo_raw
        loaded_data['delta t eval'] = delta_t_eval
        loaded_data['demonstrations raw'] = new_demo
        loaded_data['demonstrations primitive id'] = prim_id
        return loaded_data

    def fix_discontinuity_euler_space(self, loaded_data):
        return loaded_data
    '''
    def get_features_demos(self, loaded_data):
        """
        Computes useful features from demonstrations
        """
        # double cover the orientation: duplicate goals and duplicate N trajectory
        if self.augment_data_orientation:
            loaded_data = self.augment_orientation(loaded_data)
        #if self.dim_workspace == 6:
            # use tangent space mapping for address the orientation
            #self.tg_space_orient = True
            #if self.tg_space_orient:
                # generated log map
                #loaded_data = self.map_orientation_to_tangent_space(loaded_data)
                # idk why, but i get the trajectory flipped sometimes, so new fix:
                #loaded_data = self.fix_discontinuity_tg_space(loaded_data)
            #else:
            # use euler angle as reference
                #loaded_data = self.map_orientation_to_euler_angle(loaded_data)
                #loaded_data = self.fix_discontinuity_euler_space(loaded_data)


        # extract raw data
        demonstrations_raw = loaded_data['demonstrations raw']
        primitive_ids = loaded_data['demonstrations primitive id']

        # Get workspace boundaries
        x_min, x_max = self.get_workspace_boundaries(demonstrations_raw)

        # Get goal states positions
        goals = self.get_goals(demonstrations_raw, primitive_ids)

        # Normalize goals
        goals_training = normalize_state(goals, x_min, x_max)

        # Get number of demonstrated trajectories
        n_trajectories = len(demonstrations_raw)

        # Get trajectories length and eval indexes
        max_trajectory_length, trajectories_length, eval_indexes = self.get_trajectories_length(demonstrations_raw,
                                                                                                n_trajectories)

        # mask (tbh a list): demonstrations to attractor
        end_point_trajectory = np.array([demonstrations_raw[i][:, -1] for i in range(n_trajectories)])
        distance = end_point_trajectory[:, np.newaxis, :] - goals
        mask = np.argmin(np.linalg.norm(distance, axis=2), axis=1)

        if self.shaped_attractors is not None:
            # get info about the positive/negative sample for the cycle loss
            shapes = self.get_features_shapes(goals_training)
            #polar_parametrization_shapes = self.parametrize_shape_by_polar_angle(shapes[0]) if shapes is not None else None
            order_shapes, polar_parametrization_shapes = self.parametrize_shape_by_polar_angle(shapes[0])
            shapes = (
                order_shapes,  # original point set
                shapes[1]      # hard negative
            )
        else:
            #ugly fix, so sorry
            shapes = (np.zeros(10), np.zeros(10))
            polar_parametrization_shapes = np.zeros(10)

        # Collect info
        features_demos = {'x min': x_min,
                          'x max': x_max,
                          'goals': goals,
                          'goals training': goals_training,
                          'shape_attractors': shapes,
                          'orientation_parametrization_shape': polar_parametrization_shapes,
                          'trajectory 2 attractor index': mask,
                          'max demonstration length': max_trajectory_length,
                          'demonstrations length': trajectories_length,
                          'eval indexes': eval_indexes,
                          'n demonstrations': n_trajectories}

        #return loaded_data, features_demos
        return features_demos

    def parametrize_shape_by_polar_angle(self, contour):
        """
        Parameters
        ----------
        points : (N, 2) np.ndarray
            Unordered points forming a closed curve.

        Returns
        -------
        ordered_points : (N, 2) np.ndarray
            Points reordered to follow the curve.

        theta : (N,) np.ndarray
            Parameter values smoothly mapped from [-π, π].
        """
        N = len(contour)

        visited = np.zeros(N, dtype=bool)
        path = []

        tree = cKDTree(contour)

        current_idx = 0
        path.append(current_idx)
        visited[current_idx] = True

        for _ in range(N-1):
            dists, idxs = tree.query(contour[current_idx], k=N)
            for idx in idxs:
                if not visited[idx]:
                    next_idx = idx
                    break
            path.append(next_idx)
            visited[next_idx] = True
            current_idx = next_idx

        ordered_points = contour[path]
        theta = np.linspace(-np.pi, np.pi, len(ordered_points), endpoint=False)

        return ordered_points, theta


    '''
    def interpolate_points_by_line(self, goals, num_bundle=3):

        p1, p2 = goals[0, 0, :], goals[0, 1, :]
        points = np.linspace(p1, p2, num=512)
        directions = p2 - p1 / np.linalg.norm(p2-p1)
        normal = np.array([-directions[1], directions[0]])
        delta_offset, offset = 0.018, 0.0
        points_hard_negative = []
        for i in range(num_bundle):
          offset = offset + delta_offset
          offset_points_pos = points + normal * offset
          offset_points_neg = points - normal * offset
          points_hard_negative.append(offset_points_neg)
          points_hard_negative.append(offset_points_pos)

          theta = np.linspace(0, np.pi, 30)
          arc_1 = p1 + -offset*np.stack([np.cos(theta), np.sin(theta)]).T
          arc_2 = p2 + offset*np.stack([np.cos(theta), np.sin(theta)]).T
          points_hard_negative.append(arc_1)
          points_hard_negative.append(arc_2)

        hard_neg = np.concatenate(points_hard_negative)
        shapes = (points, hard_neg)

        return shapes
    '''
    def interpolate_points_by_line(self, goals, num_bundle=10, num_ellipse=2):
        p1, p2 = goals[0, 0, :], goals[0, 1, :]
        points = np.linspace(p1, p2, num=512)

        # direction & normal
        direction = p2 - p1
        direction = direction / np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])

        delta_offset, offset = 0.01, 0.0
        points_hard_negative = []

        # Ellipse params
        theta = np.linspace(0, np.pi, 60)  # finer arcs
        stretch_factors = np.linspace(1.0, 1.8, num_ellipse)  # how stretched each ellipse is
        R = np.stack([normal, direction], axis=1)  # rotation matrix

        for i in range(num_bundle):
            offset = offset + delta_offset

            # Offset lines
            offset_points_pos = points + normal * offset
            offset_points_neg = points - normal * offset
            points_hard_negative.append(offset_points_neg)
            points_hard_negative.append(offset_points_pos)

            # Add elliptical arcs for each offset level
            for factor in stretch_factors:

                a = offset                      # radius along normal
                b = offset * factor             # radius along direction
                x = a * np.cos(theta)           # normal direction (minor axis)
                y = b * np.sin(theta)           # direction (major axis)
                ellipse_local = np.stack([x, y], axis=1)  # shape (N, 2)

                # P1 side
                arc_p1 = -ellipse_local @ R.T + p1
                # P2 side (mirrored by inverting direction)
                arc_p2 = ellipse_local @ R.T + p2

                points_hard_negative.append(arc_p1)
                points_hard_negative.append(arc_p2)

        hard_neg = np.concatenate(points_hard_negative)
        shapes = (points, hard_neg)
        return shapes



    def generate_3d_circle_and_hard_negative(self, goals_training):
        p1 = goals_training[0, 0, :]
        p2 = goals_training[0, 1, :]
        c = (p1 + p2) / 2
        # CIRCLE : POSITIVE SAMPLE

        # Compute radius (distance from c to p1)
        r = np.linalg.norm(p1 - c)

        # Direction vector of the line (normalized)
        d = (p1 - c) / np.linalg.norm(p1 - c)

        # Choose an arbitrary vector not parallel to d
        a = np.array([1, 0, 0])  # Fixed initial guess
        if np.abs(np.dot(a, d)) > 0.9:  # If too parallel, switch axes
            a = np.array([0, 1, 0])

        # Compute orthogonal vectors u and v (u is perpendicular to d)
        u = np.cross(a, d)
        u = u / np.linalg.norm(u)

        # The circle lies in the plane spanned by u and d
        theta = np.linspace(0, 2 * np.pi, 1200)
        circle = c[:, None] + r * (np.cos(theta) * u[:, None] + np.sin(theta) * d[:, None])

        # TOROUS : hard negative
        # Torus parameters
        torus_minor_radius = 0.1  # Radius of the torus tube
        num_points = 1200 # Number of points to sample

        # Generate random points inside the torus
        theta_t = np.random.uniform(0, 2*np.pi, num_points)  # Angle around the circle
        phi = np.random.uniform(0, 2*np.pi, num_points)  # Angle around the tube
        radial_offset = np.random.uniform(0, torus_minor_radius, num_points)  # Random radius within tube

        # Compute the local orthonormal frame at each point on the circle
        # Tangent vector (derivative of the circle position)
        tangent = -np.sin(theta_t) * u[:, None] + np.cos(theta_t) * d[:, None]

        # Normal vector (points outward from the circle)
        normal = np.cos(theta_t) * u[:, None] + np.sin(theta_t) * d[:, None]

        # Binormal vector (perpendicular to tangent and normal)
        binormal = np.cross(tangent.T, normal.T).T
        binormal = binormal / np.linalg.norm(binormal, axis=0)  # Normalize

        # Generate torus points
        torus_points = (
            c[:, None] +
            r * (np.cos(theta_t) * u[:, None] + np.sin(theta_t) * d[:, None]) +  # Base circle
            radial_offset * np.cos(phi) * normal +  # Offset along normal
            radial_offset * np.sin(phi) * binormal   # Offset along binormal
        )

        return (circle.T, torus_points.T)

    def get_features_shapes(self, goals_training):

        if self.shaped_attractors is None:
            return None

            # target shape : [n_points, dim_workspace]

        if self.shaped_attractors == "line":
            return self.interpolate_points_by_line(goals_training)


        if self.shaped_attractors == "star":

            star = np.load('datasets/star.npy')

            # manually center the star for now
            s_max, s_min = 0.8, -0.8
            s_max_internal, s_min_internal = s_max, s_min
            s_max_external, s_min_external = s_max, s_min

            # positive sample ---> located on the limit cycle itself
            # here we just center the star in the workspace
            star_positive = s_min + (star - star.min()) * (s_max - s_min) / (star.max() - star.min())

            # negative sample ---> bundle around the positive sample
            # the number of bands around the shape is a totally free params
            star_hard_neg = []
            num_negative_bundle_around_target = 3

            # internal bundle
            for i in range(num_negative_bundle_around_target):
                s_min_internal += 0.08
                s_max_internal -= 0.08
                star_hard_neg.append(
                    s_min_internal + (star - star.min()) * (s_max_internal - s_min_internal) / (star.max() - star.min())
                )
            # external bundle
            for i in range(num_negative_bundle_around_target):
                s_min_external -= 0.08
                s_max_external += 0.08
                star_hard_neg.append(
                    s_min_external + (star - star.min()) * (s_max_external - s_min_external) / (star.max() - star.min())
                )
            star_hard_neg = np.concatenate(star_hard_neg)
            shapes = (star_positive, star_hard_neg)
            return shapes

        elif self.shaped_attractors.startswith("circle"):
            if self.dim_workspace == 3:
                return self.generate_3d_circle_and_hard_negative(goals_training)

            # 4 option:
            #       1) interpolate 2 point : center at the middle of them
            #       2) interpolate 3 point : just 1 circle
            #       3) load file containing the set of point
            #       4) just generate a random circle in the ws
            # file not present, just create it by myself for now

            if self.shaped_attractors.endswith("interpolate"):
                center, radius = self.interpolate_circle(goals_training)
            else:
                # center, radius = np.load("datasets/circle.npy")
                radius = np.array([0.3])
                center = np.zeros(self.dim_workspace)

            shapes = self.generate_hard_negative_circle(center, radius, goals_training)
            return shapes

        elif self.shaped_attractors == "sphere":
            assert self.dim_workspace > 2, "error: require 3d-attractors [sphere shaped] in 2d enviroments"
            raise NotImplementedError
            return None

        elif self.shaped_attractors in ("charmender", 'bulbasaur', 'batman', 'squirtle'):
            shape = np.load(
                os.path.join("datasets/shape", self.shaped_attractors, self.shaped_attractors+'.npy')
            )
            hard_negative = np.load(
                os.path.join("datasets/shape", self.shaped_attractors, self.shaped_attractors+'_hard_neg.npy')
            )

            max_val, min_val = 0.8, -0.8

            shape = min_val + (shape - hard_negative.min()) * (max_val - min_val) / (hard_negative.max() - hard_negative.min())
            hard_negative = min_val + (hard_negative - hard_negative.min()) * (max_val - min_val) / (hard_negative.max() - hard_negative.min())

            shapes = (shape, hard_negative)

            return shapes


    def generate_hard_negative_circle(self, center, radius, goals):
        n_points = 500
        theta = np.random.rand(n_points) * 2 * np.pi
        x_pos = center[0] + radius * np.cos(theta)
        y_pos = center[1] + radius * np.sin(theta)
        circle = np.stack((x_pos, y_pos), -1)

        num_negative_bundle_around_target = 3
        # radius of the bundle
        negative_radius = radius
        positive_radius = radius

        hard_negative = []

        for i in range(num_negative_bundle_around_target):
            theta_hard_neg = np.random.rand(n_points) * 2 * np.pi

            positive_radius = positive_radius + 0.03
            negative_radius = negative_radius - 0.03

            if negative_radius >= 0.05:  # fast fix for avoid negative circle / to small circle
                x_neg_inter = center[0] + negative_radius * np.cos(theta_hard_neg)
                y_neg_inter = center[1] + negative_radius * np.sin(theta_hard_neg)
                inter_circle = np.stack((x_neg_inter, y_neg_inter), -1)
                hard_negative.append(inter_circle)

            x_neg_ext = center[0] + positive_radius * np.cos(theta_hard_neg)
            y_neg_ext = center[1] + positive_radius * np.sin(theta_hard_neg)
            external_circle = np.stack((x_neg_ext, y_neg_ext), -1)
            hard_negative.append(external_circle)

        negative_circle = np.concatenate(hard_negative)
        shapes = (circle, negative_circle)
        return shapes

    def interpolate_circle(self, goals):
        # ps: goals structure is [multimotion, attractor_id, ws dim]
        if self.n_attractors == 2:
            center = goals[0, :, :].mean(0)
            radius = np.linalg.norm(goals[0, 0, :] - center)
        elif self.n_attractors == 3:

            a, b, c = goals[0, 0, :], goals[0, 1, :], goals[0, 2, :]
            a_size = np.linalg.norm(b-c)
            b_size = np.linalg.norm(a-c)
            c_size = np.linalg.norm(a-b)
            semi_perimeter = (a_size+ b_size + c_size)/ 2
            area = np.sqrt(semi_perimeter*(semi_perimeter-a_size)*(semi_perimeter-b_size)*(semi_perimeter-c_size))
            radius = (a_size * b_size * c_size) / (4 * area + 1e-8)  # small epsilon to avoid division by zero
            # Compute circumcenter using perpendicular bisectors
            def perp_bisector(p1, p2):
                mid = (p1 + p2) / 2
                dir_vec = p2 - p1
                perp_vec = np.array([-dir_vec[1], dir_vec[0]])
                return mid, perp_vec

            mid1, dir1 = perp_bisector(a, b)
            mid2, dir2 = perp_bisector(b, c)

            # Solve for intersection (center of circle)
            # mid1 + t1 * dir1 = mid2 + t2 * dir2  => 2 equations, 2 unknowns
            M = np.stack([dir1, -dir2], axis=1)  # shape (2, 2)
            b_vec = mid2 - mid1
            t = np.linalg.solve(M, b_vec)
            center = mid1 + t[0] * dir1

        else:
            raise Exception("number of attractor and technique for circle generation are not compatible ")
        return center, radius

    def get_workspace_boundaries(self, demonstrations_raw):
        """
        Computes workspace boundaries
        """
        if self.workspace_boundaries_type == 'from data':
            # Compute boundaries based on data
            max_single_trajectory = []
            min_single_trajectory = []

            # Get max for every trajectory in each dimension
            for j in range(len(demonstrations_raw)):
                max_single_trajectory.append(np.array(demonstrations_raw[j]).max(axis=1))
                min_single_trajectory.append(np.array(demonstrations_raw[j]).min(axis=1))

            # Get the max and min values along all of the trajectories
            x_max_orig = np.array(max_single_trajectory).max(axis=0)
            x_min_orig = np.array(min_single_trajectory).min(axis=0)

            # Add a tolerance
            x_max = x_max_orig + (x_max_orig - x_min_orig) * self.state_increment / 2
            x_min = x_min_orig - (x_max_orig - x_min_orig) * self.state_increment / 2

        elif self.workspace_boundaries_type == 'custom':
            # Use custom boundaries
            x_max = self.workspace_boundaries[:, 1]
            x_min = self.workspace_boundaries[:, 0]
        else:
            raise NameError('Selected workspace boundaries type not valid. Try: from data, custom')

        return x_min, x_max

    def get_trajectories_length(self, demonstrations_raw, n_trajectories):
        """
        Computes length trajectories, longest trajectory and evaluation indexes for fast evaluation
        """
        trajectories_length, eval_indexes = [], []
        max_trajectory_length = 0

        # Iterate through each demonstration
        for j in range(n_trajectories):
            # Get trajectory length
            length_demo = len(demonstrations_raw[j][0])
            trajectories_length.append(length_demo)

            # Find largest trajectory in demonstrations
            if length_demo > max_trajectory_length:
                max_trajectory_length = length_demo

            # Obtain indexes used for fast evaluation
            if length_demo > self.eval_length:
                eval_interval = np.floor(length_demo / self.eval_length)
                eval_indexes.append(np.arange(0, length_demo, eval_interval, dtype=np.int32))
            else:
                eval_indexes.append(np.arange(0, length_demo, 1, dtype=np.int32))

        return max_trajectory_length, trajectories_length, eval_indexes

    def get_goals(self, demonstrations_raw, primitive_ids):
        """
        Computes goal demonstrations from data --> pure goal of the end of the trajectory, no
        Info about cycle, hard negative etc for shaped attractors
        """
        # pretty sure that could be deleted:
        # @TODO --> i suspect that this function is not used anymore.
        # not sure about it. Just comment for now, in date 3/03/25. IF ever needed, delete this
        # after 1 week of this date, in general we can say that the function can be safely deleted
        # if self.dataset_name == 'cycle':
        #    return self.get_goals_limit_cycle(demonstrations_raw, primitive_ids)

        goals = []
        for i in np.unique(primitive_ids):
            ids_primitives = primitive_ids == i
            demonstrations_primitive_ids = np.array(np.where(ids_primitives))[0]
            # Iterate through trajectories of each primitive
            goals_primitive = []
            for j in demonstrations_primitive_ids:
                goals_primitive.append(np.array(demonstrations_raw[j])[:, -1])

            # Average goals and append
            if self.n_attractors == 1:
                goal_mean = np.mean(np.array(goals_primitive), axis=0)
                goals.append(goal_mean)
            else:
                goals.append(self.calc_goals_pos_via_kmeans(goals_primitive))
        return np.array(goals)

    '''
    # @TODO --> i suspect that this function is not used anymore.
    # not sure about it. Just comment for now, in date 3/03/25. IF ever needed, delete this
    # after 1 week of this date, in general we can say that the function can be safely deleted
    
    def get_goals_limit_cycle(self, demostrations_raw, primitive_ids):
        index_traj = [np.random.randint(0, demo.shape[1], self.n_attractors//len(demostrations_raw)) for demo in demostrations_raw]
        goals = [demo[:, index_traj[i]] for i, demo in enumerate(demostrations_raw)]
        goals = np.concatenate(goals, axis=1).T # [dimensionWS, n_attractor]
        return goals[np.newaxis, :, :]
    '''

    def generate_training_data(self, loaded_data, features_demos):
        """
        Normalizes demonstrations, resamples demonstrations using spline to keep a constant distance between points,
        and creates imitation window for backpropagation through time
        """
        demonstrations_raw = loaded_data['demonstrations raw']
        n_trajectories = len(demonstrations_raw)
        resampled_positions, error_acc = [], []

        # Pad demonstrations
        demonstrations_raw_padded = []
        for i in range(features_demos['n demonstrations']):
            padding_length = features_demos['max demonstration length'] - features_demos['demonstrations length'][i]
            demonstrations_raw_padded.append(np.pad(demonstrations_raw[i], ((0, 0), (0, padding_length)), mode='edge'))
        demonstrations_raw_padded = np.array(demonstrations_raw_padded)

        # Iterate through each demonstration
        for j in range(n_trajectories):
            if self.verbose:
                print('Data preprocessing, demonstration %i / %i' % (j + 1, n_trajectories))

            # Get current trajectory
            demo = np.array(demonstrations_raw_padded[j]).T
            length_demo = demo.shape[0]

            # Normalize demos
            demo_norm = normalize_state(demo, x_min=features_demos['x min'], x_max=features_demos['x max'])

            if self.spline_sample_type == 'evenly spaced':
                # Create phase array that spatially parametrizes demo in one dimension
                curve_phase = 0
                curve_phases, delta_phases = [curve_phase], []

                for i in range(length_demo - 1):  # iterate through every point in trajectory and assign a phase value
                    # Compute phase increment based on distance of consecutive points
                    delta_phase = np.linalg.norm(demo_norm[i + 1, :] - demo_norm[i, :])

                    if delta_phase == 0:
                        # If points in trajectory have zero phase difference, splprep throws error -> add small margin
                        delta_phase += 1e-15

                    # Increment phase
                    curve_phase += delta_phase

                    # Store phase and delta of current point in curve
                    curve_phases.append(curve_phase)
                    delta_phases.append(delta_phase)

                delta_phases.append(0)  # zero delta for last point
                curve_phases = np.array(curve_phases)
                delta_phases = np.array(delta_phases)
                max_phase = curve_phases[-1]

            elif self.spline_sample_type == 'from data' or self.spline_sample_type == 'from data resample':
                curve_phases = np.arange(0, length_demo * self.delta_t, self.delta_t)
                delta_phases = np.ones(length_demo) * self.delta_t  # TODO: could be extended to variable delta t
                max_phase = np.max(curve_phases)
            else:
                raise NameError('Spline sample type not valid, check params file for options.')

            # Create input for spline: demonstrations and corresponding phases
            spline_input = []
            for i in range(self.dim_workspace):
                spline_input.append(demo_norm[:, i])
            spline_input.append(curve_phases)
            spline_input.append(delta_phases)

            # Fit spline
            spline_parameters, _ = splprep(spline_input, s=0, k=1, u=curve_phases)  # s = 0 -> no smoothing; k = 1 -> linear interpolation

            # Create initial phases u with spatially equidistant points
            if self.spline_sample_type == 'evenly spaced' or self.spline_sample_type == 'from data resample':
                u = np.linspace(0, max_phase, self.trajectories_resample_length)
            elif self.spline_sample_type == 'from data':
                u = curve_phases
            else:
                raise NameError('Spline sample type not valid, check params file for options.')

            # Iterate using imitation window size to get position labels for backpropagation through time
            window = []
            for _ in range(self.imitation_window_size + (self.dynamical_system_order - 1)):
                # Compute demo positions based on current phase value
                spline_values = splev(u, spline_parameters)
                position_window = spline_values[:self.dim_workspace]

                # Append position to window trajectory
                window.append(position_window)

                # Find time/phase for next point in imitation window
                delta_phase = spline_values[-1]
                next_t = u + delta_phase
                u = np.clip(next_t, a_min=0, a_max=max_phase)  # update phase

                # Accumulate error for debugging
                predicted_phase = splev(u, spline_parameters)[-2]
                error_acc.append(np.mean(np.abs(predicted_phase - u)))

            resampled_positions.append(window)

        if self.verbose:
            print('Mean error spline resampling:', np.mean(error_acc))

        # Change axes order to one more intuitive
        # 0: trajectories; 1: states trajectory; 2: state dimensions; 3: imitation window position
        resampled_positions = np.transpose(np.array(resampled_positions), (0, 3, 2, 1))
        return resampled_positions

    def get_limits_derivatives(self, demos):
        """
        Computes velocity and acceleration of the training demonstrations
        """
        # Get velocities from normalized resampled demonstrations
        velocity = (demos[:, :, :, 1:] - demos[:, :, :, :-1]) / self.delta_t

        # Get accelerations from velocities
        acceleration = (velocity[:, :, :, 1:] - velocity[:, :, :, :-1]) / self.delta_t

        # Compute max velocities
        min_velocity = np.min(velocity, axis=(0, 1, 3))
        max_velocity = np.max(velocity, axis=(0, 1, 3))

        # Compute max acceleration
        if self.dynamical_system_order == 1:
            min_acceleration = None  # acceleration not used in first-order systems
            max_acceleration = None
        elif self.dynamical_system_order == 2:
            min_acceleration = np.min(acceleration, axis=(0, 1, 3))
            max_acceleration = np.max(acceleration, axis=(0, 1, 3))
        else:
            raise ValueError('Selected dynamical system order not valid, options: 1, 2.')

        # If second order, since the velocity is part of the state, we extend its limits
        if self.dynamical_system_order == 2:
            max_velocity_state = max_velocity + (max_velocity - min_velocity) * self.state_increment / 2
            min_velocity_state = min_velocity - (max_velocity - min_velocity) * self.state_increment / 2
            max_velocity = max_velocity_state
            min_velocity = min_velocity_state

        # Collect
        limits = {'vel min train': min_velocity,
                  'vel max train': max_velocity,
                  'acc min train': min_acceleration,
                  'acc max train': max_acceleration}
        return limits

    def calc_goals_pos_via_kmeans(self, goals_primitive) -> list:
        """
        Computes goal position using kmeans
        """
        kmeans = KMeans(n_clusters=self.n_attractors, random_state=0).fit(goals_primitive)
        return kmeans.cluster_centers_.tolist()
