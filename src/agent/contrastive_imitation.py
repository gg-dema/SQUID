import numpy as np
import torch
import torch.nn.functional as F


from agent.neural_network import NeuralNetwork
from agent.utils.ranking_losses import (
    ContrastiveLoss,
    TripletLoss,
    ContrastiveLossSphericalSpace,
    ContrastiveLossNorm,
    TripletAngleLoss,
    great_circle_distance,
    ManifoldBased_ContrastiveLoss
)
from agent.dynamical_system import DynamicalSystem
from agent.utils.dynamical_system_operations import normalize_state



class ContrastiveImitation:
    """
    Computes SUID losses and optimizes Neural Network
    """
    def __init__(self, data, params):
        # Params file parameters

        # DYNAMICAL SYSTEM PARAMETERS
        # ---------------------------
        self.dim_workspace = params.workspace_dimensions
        self.spherical_latent_space = params.spherical_latent_space
        self.dynamical_system_order = params.dynamical_system_order
        self.dim_state = self.dim_workspace
        self.dim_state *= self.dynamical_system_order
        self.multi_motion = params.multi_motion
        self.latent_dyn_type = params.latent_dynamic_system_type                # specify which dynamical system we want to use
        self.latent_dimension = params.latent_space_dim                         # dimension of the latent space system
        # boundary of the output of the model
        self.min_vel = torch.from_numpy(data['vel min train'].reshape([1, self.dim_workspace])).float().cuda()
        self.max_vel = torch.from_numpy(data['vel max train'].reshape([1, self.dim_workspace])).float().cuda()
        if data['acc min train'] is not None:
            min_acc = torch.from_numpy(data['acc min train'].reshape([1, self.dim_workspace])).float().cuda()
            max_acc = torch.from_numpy(data['acc max train'].reshape([1, self.dim_workspace])).float().cuda()
        else:
            min_acc = None
            max_acc = None

        # boundary of the input space --> used for normalize the input
        self.x_min, self.x_max = data['x min'], data['x max']
        self.params_dynamical_system = {'saturate transition': params.saturate_out_of_boundaries_transitions,
                                        'x min': data['x min'],
                                        'x max': data['x max'],
                                        'vel min train': self.min_vel,
                                        'vel max train': self.max_vel,
                                        'acc min train': min_acc,
                                        'acc max train': max_acc}


        # TRAINING PARAMETERS
        # -------------------
        self.imitation_window_size = params.imitation_window_size
        self.batch_size = params.batch_size
        self.batch_size_contrastive_norm = params.batch_size_contrastive_norm
        self.batch_size_orientation_loss = params.batch_size_orientation_loss
        self.generalization_window_size = params.stabilization_window_size      # n step of integration on the latent space
        self.imitation_loss_weight = params.imitation_loss_weight               # n step of integration on the task space
        self.stabilization_loss = params.stabilization_loss                     # specify which kind of stable loss we want to use
        self.save_path = params.results_path

        # loss weight
        self.boundary_loss_weight = params.boundary_loss_weight
        self.stabilization_loss_weight = params.stabilization_loss_weight
        self.goal_mapping_loss_weight = params.goal_mapping_loss_weight
        self.cycle_loss_weight = params.cycle_loss_weight
        self.cycle_orientation_loss_weight = params.cycle_orientation_loss_weight
        self.goal_mapping_decreasing_weight = params.goal_mapping_decreasing_weight

        self.load_model = params.load_model
        self.results_path = params.results_path
        self.interpolation_sigma = params.interpolation_sigma
        self.delta_t = 1  # used for training, can be anything

        # Parameters data processor
        self.primitive_ids = np.array(data['demonstrations primitive id'])
        self.n_primitives = data['n primitives']
        self.goals_tensor = torch.FloatTensor(data['goals training']).cuda()
        self.demonstrations_train = data['demonstrations train']
        self.n_demonstrations = data['n demonstrations']
        self.demonstrations_length = data['demonstrations length']

        # GOAL data --> specify info about the latent/task goal
        self.no_goals_set_yet = True
        self.reference_goals = None
        self.attractor_shape = data['shape_attractors']                               # tuple of np.array (contour, hard negative) -> None if not used
        self.attractor_shape_orientation = data['orientation_parametrization_shape']  # parametrization of the point on the continuous curve
        self.n_attractors = params.n_attractors

        # INIT NEURAL MODEL
        # -----------------
        self.mse_loss = torch.nn.MSELoss()
        self.triplet_loss = TripletLoss(margin=params.triplet_margin, swap=True)

        if self.stabilization_loss == "triplet":
            self.triplet_loss = TripletAngleLoss(margin=params.triplet_margin)

        elif self.stabilization_loss == 'contrastive':
            if self.spherical_latent_space:
                self.contrastive_loss = ManifoldBased_ContrastiveLoss(margin=params.contrastive_margin,
                                                                      latent_manifold=self.latent_dyn_type)

                self.contrastive_loss = ContrastiveLossSphericalSpace(margin=params.contrastive_margin)
            else:
                self.contrastive_loss = ContrastiveLoss(margin=params.contrastive_margin)

        if self.attractor_shape:
            self.contrastive_norm_loss = ContrastiveLossNorm(margin=params.contrastive_norm_margin)
        # Initialize Neural Network
        self.model = NeuralNetwork(dim_state=self.dim_state,
                                   dynamical_system_order=self.dynamical_system_order,
                                   n_primitives=self.n_primitives,
                                   multi_motion=self.multi_motion,
                                   latent_gain_lower_limit=params.latent_gain_lower_limit,
                                   latent_gain_upper_limit=params.latent_gain_upper_limit,
                                   latent_gain=params.latent_gain,
                                   latent_space_dim=params.latent_space_dim,
                                   neurons_hidden_layers=params.neurons_hidden_layers,
                                   adaptive_gains=params.adaptive_gains,
                                   n_attractors=self.n_attractors,
                                   latent_system_dynamic_type=params.latent_dynamic_system_type,
                                   sigma=params.sigma,
                                   ).cuda()

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=params.learning_rate,
                                           weight_decay=params.weight_decay)

        # Load Neural Network if requested
        if self.load_model:
            self.model.load_state_dict(torch.load(self.results_path + 'model'), strict=False)

        # Initialize latent goals
        self.model.update_goals_latent_space(self.goals_tensor)
        if self.latent_dyn_type == "limit cycle":
            self.model.update_goals_shape(self.attractor_shape[0])

    def init_dynamical_system(self, initial_states, primitive_type=None, delta_t=1):
        """
        Creates dynamical system using the parameters/variables of the learning policy
        """
        # If no primitive type, assume single-model learning
        if primitive_type is None:
            primitive_type = torch.FloatTensor([1])

        # Create dynamical system
        dynamical_system = DynamicalSystem(x_init=initial_states,
                                           model=self.model,
                                           primitive_type=primitive_type,
                                           order=self.dynamical_system_order,
                                           min_state_derivative=[self.params_dynamical_system['vel min train'],
                                                                 self.params_dynamical_system['acc min train']],
                                           max_state_derivative=[self.params_dynamical_system['vel max train'],
                                                                 self.params_dynamical_system['acc max train']],
                                           saturate_transition=self.params_dynamical_system['saturate transition'],
                                           dim_state=self.dim_state,
                                           delta_t=delta_t,
                                           x_min=self.params_dynamical_system['x min'],
                                           x_max=self.params_dynamical_system['x max'],
                                           spherical_latent_space=self.spherical_latent_space,
                                           latent_dyn_type=self.latent_dyn_type)

        return dynamical_system

    def imitation_cost(self, state_sample, primitive_type_sample):
        """
        Imitation cost: MSE between the generated trajectories and the references
        """
        # Create dynamical system
        dynamical_system = self.init_dynamical_system(initial_states=state_sample[:, :, 0],
                                                      primitive_type=primitive_type_sample)

        # Compute imitation error for transition window
        imitation_error_accumulated = 0

        for i in range(self.imitation_window_size - 1):
            # Compute transition
            x_t_d = dynamical_system.transition(space='task')['desired state']

            # Compute and accumulate error
            imitation_error_accumulated += self.mse_loss(x_t_d[:, :self.dim_workspace], state_sample[:, :self.dim_workspace, i + 1].cuda())

        # normalize the error across the different time step
        imitation_error_accumulated = imitation_error_accumulated / (self.imitation_window_size - 1)

        return imitation_error_accumulated * self.imitation_loss_weight

    def contrastive_matching(self, state_sample, primitive_type_sample):
        """
        Transition matching cost : latent space dynamic learning
        """
        # Create dynamical systems
        dynamical_system_task = self.init_dynamical_system(initial_states=state_sample,
                                                           primitive_type=primitive_type_sample)

        dynamical_system_latent = self.init_dynamical_system(initial_states=state_sample,
                                                             primitive_type=primitive_type_sample)

        # Compute cost over trajectory
        contrastive_matching_cost = 0
        batch_size = state_sample.shape[0]

        for i in range(self.generalization_window_size):
            # Do transition
            y_t_task_prev = dynamical_system_task.y_t['task']
            y_t_task = dynamical_system_task.transition(space='task')['latent state']
            _, y_t_latent = dynamical_system_latent.transition_latent_system()

            if i > 0:  # we need at least one iteration to have a previous point to push the current one away from
                # Transition matching cost
                if self.stabilization_loss == 'contrastive':
                    # Anchor
                    anchor_samples = torch.cat((y_t_task, y_t_task))

                    # Positive/Negative samples
                    contrastive_samples = torch.cat((y_t_latent, y_t_task_prev))

                    # Contrastive label
                    contrastive_label_pos = torch.ones(batch_size).cuda()
                    contrastive_label_neg = torch.zeros(batch_size).cuda()
                    contrastive_label = torch.cat((contrastive_label_pos, contrastive_label_neg))

                    # Compute cost
                    contrastive_matching_cost += self.contrastive_loss(anchor_samples, contrastive_samples, contrastive_label)

                elif self.stabilization_loss == 'triplet':
                    contrastive_matching_cost += self.triplet_loss(y_t_task, y_t_latent, y_t_task_prev)
        contrastive_matching_cost = contrastive_matching_cost / (self.generalization_window_size - 1)

        return contrastive_matching_cost * self.stabilization_loss_weight

    def goal_mapping(self, primitive_type_sample):
        """
        goal mapping loss: used for let the model learn a specific representation of the goal points
        """

        goals = self.goals_tensor

        goals_2nd_order = torch.zeros([self.n_primitives, self.n_attractors, self.dim_state])
        goals_2nd_order[:, :, :self.dim_workspace] = goals
        goals_latent_space = self.model.encoder(goals_2nd_order, primitive_type_sample)  # 1 2 300


        # @TODO: fix the different selection of the goal for different n of attractors
        if self.n_attractors == 2:
            reference_goals = torch.ones_like(goals_latent_space[0, :, :])
            reference_goals[1, :] *= -1

        elif self.n_attractors > 2:
            if self.no_goals_set_yet:
                # we can just sample some goal point. If we didn't have any fixed representation yet, here we sample the latent space
                self.reference_goals = torch.rand(goals_latent_space[0, :, :].shape).cuda()
                self.reference_goals = torch.nn.functional.normalize(self.reference_goals) if self.spherical_latent_space else self.reference_goals
                reference_goals = self.reference_goals
                self.no_goals_set_yet = False
            else:
                reference_goals = self.reference_goals
        else:
             raise ValueError("not supported number of attractors")

        if self.dynamical_system_order == 2:
            reference_goals[:, :self.latent_dimension//2] = 0.0

        if self.spherical_latent_space:
            # calculate the distance between the goals and the reference point directly on the manifold
            # AKA use directly the great_circle_distance
            loss = [
                0.5*torch.sum(great_circle_distance(goals_latent_space[0, i, :], reference_goals[i, :], dim=0))**2
                for i in range(self.n_attractors)
            ]
        else:
            loss = [
                torch.nn.functional.mse_loss(goals_latent_space[0, i, :], reference_goals[i, :]) for i in range(self.n_attractors)
            ]

        if self.goal_mapping_loss_weight > 1e-10:
            self.goal_mapping_loss_weight *= self.goal_mapping_decreasing_weight

        return self.goal_mapping_loss_weight * (sum(loss))

    def limit_cycle_loss(self, primitive_type_sample):
        """
        Limit cycle loss: impose that the norm of the first 2 element of the latent state is equal to one
        This use a contrastive loss, with a set of hard negative defined as the point near to the desired attraction curve
        """

        # ps: this for now consider just 2d cases: what about 3d?
        positive_sample = self.attractor_shape[0]
        negative_sample = self.attractor_shape[1]

        # select random point id on the curve ---> positive sample
        selected_pos_id = np.random.choice(positive_sample.shape[0],
                                           size=self.batch_size_contrastive_norm,
                                           replace=False)  # replace do not allow for duplicate in the sample

        # select random point id between the hard negative set --> negative sample
        selected_neg_id = np.random.choice(negative_sample.shape[0],
                                           size=self.batch_size_contrastive_norm,
                                           replace=False)

        # convert the desired point into torch tensor
        selected_pos = torch.FloatTensor(positive_sample[selected_pos_id, :]).cuda()
        selected_neg = torch.FloatTensor(negative_sample[selected_neg_id, :]).cuda()

        # add random point in the grid for negative sample (simple negative, not only hard negative)
        random_point = torch.rand(selected_neg.shape).cuda()

        # @LEGACY COMPARISON : this code under comment is used for test the effect of remove the hard point from the loss formulation
        # and it's keep for possible comparison
        # @LEGACY COMPARISON : random_point = torch.rand((self.batch_size_contrastive_norm, 2)).cuda()

        # project all the selected point into the latent space
        latent_positive = self.model.encoder(selected_pos, primitive_type_sample).unsqueeze(0)
        latent_negative = self.model.encoder(selected_neg, primitive_type_sample).unsqueeze(0)
        latent_random_point = self.model.encoder(random_point, primitive_type_sample).unsqueeze(0)

        # create the label for the contrastive loss: 1 for positive, 0 for negative
        negative_label = torch.zeros(latent_negative.shape[1] + latent_random_point.shape[1])
        # @LEGACY COMPARISON : negative_label = torch.zeros(latent_random_point.shape[1])
        positive_label = torch.ones(latent_positive.shape[1])

        x = torch.concatenate([latent_positive, latent_negative, latent_random_point], axis=1)
        # @LEGACY COMPARISON : x = torch.concatenate([latent_positive, latent_random_point], axis=1)
        target = torch.concatenate([positive_label, negative_label]).cuda()
        return self.contrastive_norm_loss(x, target)

    def limit_cycle_loss_orient(self, primitive_type):
        """
        orientation loss : MSE between the second element of the latent state (after the change to polar coordinate) and the reference orientation
        """

        def angle_loss(pred_angle, target_angle):
            # parametrize angle as [sin(angle), cos(angle)] vector for address singular representation in orientation
            pred_vec = torch.stack([torch.cos(pred_angle), torch.sin(pred_angle)], dim=-1)
            target_vec = torch.stack([torch.cos(target_angle),  torch.sin(target_angle)], dim=-1)
            # return mse loss
            return F.mse_loss(pred_vec, target_vec)

        # @LEGACY CODE : different implementation of the angle loss --> contrastive formulation
        #def contrastive_angle(pred_angle, target_angle, margin=1e-10):             # @TODO bad comment: fix context around the folder of test experiment
            #
            # margin=0.03
            #phi = pred_angle - target_angle
            #normalized_error = torch.remainder(phi + torch.pi, 2*torch.pi) - torch.pi
        #    pred_vec = torch.stack([torch.cos(pred_angle), torch.sin(pred_angle)], dim=-1)
        #    target_vec = torch.stack([torch.cos(target_angle),  torch.sin(target_angle)], dim=-1)

        #    return torch.clip(
                #torch.abs(normalized_error) - margin,
        #        torch.abs(pred_vec - target_vec) - margin,
        #        min=0.0,
        #        max=None
        #    ).mean()

        # Select control point
        id = np.random.choice(self.attractor_shape[0].shape[0],
                              size=self.batch_size_orientation_loss,
                              replace=False)


        # convert to cuda the point on the reference shape
        attractor_shape = torch.FloatTensor(self.attractor_shape[0][id, :]).cuda()
        # convert to cuda the orientation of such point
        orient_shape = torch.FloatTensor(self.attractor_shape_orientation[id]).cuda()

        # project to the latent space and calc orientation
        latent_state = self.model.encoder(attractor_shape, primitive_type)
        orient_latent_state_polar = torch.arctan2(latent_state[..., 0], latent_state[..., 1])

        return angle_loss(orient_latent_state_polar, orient_shape)
        # LEGACY CODE
        #return contrastive_angle(orient_latent_state_polar, orient_shape)

        #error = (orient_shape - orient_latent_state_polar)
        #normalized_error = torch.remainder(error + torch.pi, 2*torch.pi) - torch.pi
        #return 0.5 * normalized_error.pow(2).mean()

        #return F.mse_loss(orient_shape, orient_latent_state_polar)

    def boundary_constrain(self, state_sample, primitive_type_sample):
        """ use as loss the cosine similarity between the vector at boundary and the orthogonal vector of the boundary"""

        # Force states to start at the boundary
        selected_axis = torch.randint(low=0, high=self.dim_state, size=[self.batch_size])
        selected_limit = torch.randint(low=0, high=2, size=[self.batch_size])
        limit_options = torch.FloatTensor([-1, 1])
        limits = limit_options[selected_limit]
        replaced_samples = torch.arange(start=0, end=self.batch_size)
        state_sample[replaced_samples, selected_axis] = limits.cuda()

        # Create dynamical systems
        self.params_dynamical_system['saturate transition'] = False
        dynamical_system = self.init_dynamical_system(initial_states=state_sample,
                                                      primitive_type=primitive_type_sample)
        self.params_dynamical_system['saturate transition'] = True

        # Do one transition at the boundary and get velocity
        transition_info = dynamical_system.transition()
        x_t_d = transition_info['desired state']
        dx_t_d = transition_info['desired velocity']

        # Iterate through every dimension
        epsilon = 5e-2
        loss = 0
        states_boundary = self.dim_workspace
        for i in range(states_boundary):
            distance_upper = torch.abs(x_t_d[:, i] - 1)
            distance_lower = torch.abs(x_t_d[:, i] + 1)

            # Get velocities for points in the boundary
            dx_axis_upper = dx_t_d[distance_upper < epsilon]
            dx_axis_lower = dx_t_d[distance_lower < epsilon]

            # Compute normal vectors for lower and upper limits
            normal_upper = torch.zeros(dx_axis_upper.shape).cuda()
            normal_upper[:, i] = 1
            normal_lower = torch.zeros(dx_axis_lower.shape).cuda()
            normal_lower[:, i] = -1

            # Compute dot product between boundary velocities and normal vectors
            dot_product_upper = torch.bmm(dx_axis_upper.view(-1, 1, self.dim_workspace),
                                          normal_upper.view(-1, self.dim_workspace, 1)).reshape(-1)

            dot_product_lower = torch.bmm(dx_axis_lower.view(-1, 1, self.dim_workspace),
                                          normal_lower.view(-1, self.dim_workspace, 1)).reshape(-1)

            # Concat with zero in case no points sampled in boundaries, to avoid nans
            dot_product_upper = torch.cat([dot_product_upper, torch.zeros(1).cuda()])
            dot_product_lower = torch.cat([dot_product_lower, torch.zeros(1).cuda()])

            # Compute losses
            loss += F.relu(dot_product_upper).mean()
            loss += F.relu(dot_product_lower).mean()

        loss = loss / (2 * self.dim_workspace)

        return loss * self.boundary_loss_weight

    def demo_sample(self):
        """
        Samples a batch of windows from the demonstrations
        """

        # Select demonstrations randomly
        selected_demos = np.random.choice(range(self.n_demonstrations), self.batch_size)

        # Get random points inside trajectories
        i_samples = []
        for i in range(self.n_demonstrations):
            selected_demo_batch_size = sum(selected_demos == i)
            demonstration_length = self.demonstrations_train.shape[1]
            i_samples = i_samples + list(np.random.randint(0, demonstration_length, selected_demo_batch_size, dtype=int))

        # Get sampled positions from training data
        position_sample = self.demonstrations_train[selected_demos, i_samples]
        position_sample = torch.FloatTensor(position_sample).cuda()

        # Create empty state
        state_sample = torch.empty([self.batch_size, self.dim_state, self.imitation_window_size]).cuda()

        # Fill first elements of the state with position
        state_sample[:, :self.dim_workspace, :] = position_sample[:, :, (self.dynamical_system_order - 1):]

        # Fill rest of the elements with velocities for second order systems
        if self.dynamical_system_order == 2:
            velocity = (position_sample[:, :, 1:] - position_sample[:, :, :-1]) / self.delta_t
            velocity_norm = normalize_state(velocity,
                                            x_min=self.min_vel.reshape(1, self.dim_workspace, 1),
                                            x_max=self.max_vel.reshape(1, self.dim_workspace, 1))
            state_sample[:, self.dim_workspace:, :] = velocity_norm

        # Finally, get primitive ids of sampled batch (necessary when multi-motion learning)
        primitive_type_sample = self.primitive_ids[selected_demos]
        primitive_type_sample = torch.FloatTensor(primitive_type_sample).cuda()

        return state_sample, primitive_type_sample

    def space_sample(self):
        """
        Samples a batch of windows from the state space
        """
        with torch.no_grad():
            # Sample state
            state_sample_gen = torch.Tensor(self.batch_size, self.dim_state).uniform_(-1, 1).cuda()

            # Choose sampling methods
            if not self.multi_motion:
                primitive_type_sample_gen = torch.randint(0, self.n_primitives, (self.batch_size,)).cuda()
            else:
                # If multi-motion learning also sample in interpolation space
                # sigma of the samples are in the demonstration spaces
                encodings = torch.eye(self.n_primitives).cuda()
                primitive_type_sample_gen_demo = encodings[torch.randint(0, self.n_primitives, (round(self.batch_size * self.interpolation_sigma),)).cuda()]

                # 1 - sigma  of the samples are in the interpolation space
                primitive_type_sample_gen_inter = torch.rand(round(self.batch_size * (1 - self.interpolation_sigma)), self.n_primitives).cuda()

                # Concatenate both samples
                primitive_type_sample_gen = torch.cat((primitive_type_sample_gen_demo, primitive_type_sample_gen_inter), dim=0)

        return state_sample_gen, primitive_type_sample_gen

    def compute_loss(self, state_sample_IL, primitive_type_sample_IL, state_sample_gen, primitive_type_sample_gen):
        """
        Computes total cost
        """
        loss_list = []  # list of losses
        losses_names = []

        # Learning from demonstrations outer loop
        if self.imitation_loss_weight != 0:
            imitation_cost = self.imitation_cost(state_sample_IL, primitive_type_sample_IL)
            loss_list.append(imitation_cost)
            losses_names.append('Imitation')

        # Transition matching
        if self.stabilization_loss_weight != 0:
            contrastive_matching_cost = self.contrastive_matching(state_sample_gen, primitive_type_sample_gen)
            loss_list.append(contrastive_matching_cost)
            losses_names.append('Stability')

        if self.goal_mapping_loss_weight > 1e-10:
            goal_mapping_cost = self.goal_mapping(primitive_type_sample_gen)
            loss_list.append(goal_mapping_cost)
            losses_names.append("Goal mapping")

        if self.boundary_loss_weight != 0:
            state_sample_gen_clone = torch.clone(state_sample_gen)
            boundary_loss = self.boundary_constrain(state_sample_gen_clone, primitive_type_sample_gen)
            loss_list.append(boundary_loss)
            losses_names.append('boundary')

        if self.cycle_loss_weight != 0:
            loss_cycle = self.cycle_loss_weight * self.limit_cycle_loss(primitive_type_sample_gen)
            loss_list.append(loss_cycle)
            losses_names.append('cycle')

        if self.cycle_orientation_loss_weight != 0:
            loss_cycle_orient = self.cycle_orientation_loss_weight * self.limit_cycle_loss_orient(primitive_type_sample_gen)
            loss_list.append(loss_cycle_orient)
            losses_names.append('angle')

        # Sum losses
        loss = 0
        for i in range(len(loss_list)):
            loss += loss_list[i]

        return loss, loss_list, losses_names

    def update_model(self, loss):
        """
        Updates Neural Network with computed cost
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update goal in latent space
        self.model.update_goals_latent_space(self.goals_tensor)
        if self.latent_dyn_type == "limit cycle":
            self.model.update_goals_shape(self.attractor_shape[0])

    def train_step(self):
        """
        Samples data and trains Neural Network
        """
        # Sample from space
        state_sample_gen, primitive_type_sample_gen = self.space_sample()

        # Sample from trajectory
        state_sample_IL, primitive_type_sample_IL = self.demo_sample()

        # Get loss from CONDOR
        loss, loss_list, losses_names = self.compute_loss(state_sample_IL,
                                                          primitive_type_sample_IL,
                                                          state_sample_gen,
                                                          primitive_type_sample_gen)

        # Update model
        self.update_model(loss)

        return loss, loss_list, losses_names






