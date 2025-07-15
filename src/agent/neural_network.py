import torch
import numpy as np
import agent.utils.polar as polar

import itertools

class NeuralNetwork(torch.nn.Module):
    """
    Neural Network model
    """
    def __init__(self, dim_state, dynamical_system_order, n_primitives, multi_motion, latent_gain_lower_limit,
                 latent_gain_upper_limit, latent_gain, latent_space_dim, neurons_hidden_layers, adaptive_gains,
                 n_attractors, latent_system_dynamic_type, sigma):
        super(NeuralNetwork, self).__init__()
        
        # Initialize Network parameters
        self.n_input = dim_state
        n_output = dim_state // dynamical_system_order
        self.multi_motion = multi_motion
        self.n_primitives = n_primitives
        self.latent_space_dim = latent_space_dim
        self.dynamical_system_order = dynamical_system_order
        self.latent_gain = latent_gain
        self.latent_gain_lower_limit = latent_gain_lower_limit
        self.latent_gain_upper_limit = latent_gain_upper_limit
        self.latent_gain_lower_limit_beta = 3.0
        self.latent_gain_upper_limit_beta = 3.5
        self.adaptive_gains = adaptive_gains
        self.n_attractors = n_attractors
        self.latent_system_dynamic_type = latent_system_dynamic_type
        self.sigma = sigma

        # Select activation function
        self.activation = torch.nn.GELU()
        self.sigmoid = torch.nn.Sigmoid()

        # Initialize goals list
        self.goals_latent_space = [list(np.zeros(n_attractors)) for _ in range(n_primitives)]
        self.y_g_avg = torch.zeros(self.latent_space_dim).cuda()       # avg goal ---> mean point between all the other goals. Obtained projection phi(x) = y and then by y.mean
        # Primitives encodings
        self.primitives_encodings = torch.eye(n_primitives).cuda()

        # Initialize encoder layers: psi
        if multi_motion:
            self.encoder1 = torch.nn.Linear(self.n_input + n_primitives, neurons_hidden_layers)
        else:
            self.encoder1 = torch.nn.Linear(self.n_input, neurons_hidden_layers)
        self.norm_e_1 = torch.nn.LayerNorm(neurons_hidden_layers)
        self.encoder2 = torch.nn.Linear(neurons_hidden_layers, neurons_hidden_layers)
        self.norm_e_2 = torch.nn.LayerNorm(neurons_hidden_layers)
        self.encoder3 = torch.nn.Linear(neurons_hidden_layers, neurons_hidden_layers)
        self.norm_e_3 = torch.nn.LayerNorm(neurons_hidden_layers)

        self.encoder4 = torch.nn.Linear(neurons_hidden_layers, self.latent_space_dim)
        self.norm_e_4 = torch.nn.LayerNorm(neurons_hidden_layers)

        # Norm output latent space
        self.norm_latent = torch.nn.LayerNorm(self.latent_space_dim)

        # Initialize dynamical system decoder layers: phi
        self.decoder1_dx = torch.nn.Linear(self.latent_space_dim, neurons_hidden_layers)
        self.norm_de_dx1 = torch.nn.LayerNorm(neurons_hidden_layers)
        self.decoder2_dx = torch.nn.Linear(neurons_hidden_layers, neurons_hidden_layers)
        self.norm_de_dx2 = torch.nn.LayerNorm(neurons_hidden_layers)
        self.decoder3_dx = torch.nn.Linear(neurons_hidden_layers, neurons_hidden_layers)
        self.norm_de_dx3 = torch.nn.LayerNorm(neurons_hidden_layers)
        self.decoder4_dx = torch.nn.Linear(neurons_hidden_layers, n_output)

        # Latent space
        self.gain_nn_1 = torch.nn.Linear(self.latent_space_dim, self.latent_space_dim)

        self.norm_latent_gain_input = torch.nn.LayerNorm(self.latent_space_dim)
        self.norm_gain_1 = torch.nn.LayerNorm(self.latent_space_dim)
        self.gain_nn_2 = self.select_output_gain_nn(self.latent_space_dim)  # different shape based on different latent dynamic

        self.gain_nn_beta_1 = torch.nn.Linear(self.latent_space_dim, self.latent_space_dim)
        self.norm_gain_beta = torch.nn.LayerNorm(self.latent_space_dim)
        self.gain_nn_beta_2 = torch.nn.Linear(self.latent_space_dim, 1)

        self.norm_de_dx0 = []
        for i in range(self.dynamical_system_order):
            self.norm_de_dx0.append(torch.nn.LayerNorm(self.latent_space_dim))

        self.norm_de_dx0 = torch.nn.ModuleList(self.norm_de_dx0)
        self.norm_de_dx0_0 = torch.nn.LayerNorm(self.latent_space_dim)
        self.norm_de_dx0_1 = torch.nn.LayerNorm(self.latent_space_dim)


        self.R = torch.eye(self.latent_space_dim)
        self.R[0, 0] = 0
        self.R[0, 1] = -1
        self.R[1, 0] = 1
        self.R[1, 1] = 0
        self.R = self.R.cuda()

    def select_output_gain_nn(self, latent_input_size):

        if self.latent_system_dynamic_type in ("standard", "gaussian", "gravity"):
            return torch.nn.Linear(self.latent_space_dim, latent_input_size)
        elif self.latent_system_dynamic_type in ("multivariate", "multivariate_s2", "torus"):            # multiple gain for each equation of the latent state ---> unbalance behaviour, keep for now
            # a single alpha parameters for  (if use it back, change the dim of alpha --> look at multivariate dynamics)
            # return torch.nn.Linear(self.latent_space_dim, latent_input_size*self.n_attractors)
            return torch.nn.Linear(self.latent_space_dim, latent_input_size)
        elif self.latent_system_dynamic_type == "well known":
            raise NotImplementedError(' well known still to define ==> gain nn for alpha')
        elif self.latent_system_dynamic_type in ('limit cycle', 'limit cycle avg'):
            # return torch.nn.Linear(self.latent_space_dim, latent_input_size - 2)
            return torch.nn.Linear(self.latent_space_dim, latent_input_size)
        else:
            raise NameError("bad latent dynamic selected")

    def update_goals_latent_space(self, goals):
        """
        Maps task space goal to latent space goal
        """
        # if single attractor, expand the dimension for match calc
        if goals.ndim == 2:
            goals = goals.unsqueeze(0)

        for i in range(self.n_primitives):
            for attractor_id in range(self.n_attractors):
                primitive_type = torch.FloatTensor([i]).cuda()
                input = torch.zeros([1, self.n_input])  # add zeros as velocity goal for second order DS
                input[:, :goals[i][attractor_id].shape[0]] = goals[i][attractor_id]
                self.goals_latent_space[i][attractor_id] = self.encoder(input, primitive_type)

    def update_goals_shape(self, goals_shape):
        """
            maps the task shape contour to a latent space representation,
            then, compute the part used in the linear dynamic (the avg goals shape)
        """
        primitive_type = torch.FloatTensor([0]).cuda()
        goals_shape = torch.FloatTensor(goals_shape).cuda()

        self.y_g_avg = self.encoder(goals_shape, primitive_type).mean(0)

    def get_goals_latent_space_batch(self, primitive_type):
        """
        Creates a batch with latent space goals computed in 'update_goals_latent_space'
        """
        goals_latent_space_batch = torch.zeros(primitive_type.shape[0],
                                               self.latent_space_dim,
                                               self.n_attractors).cuda()
        for i in range(self.n_primitives):
            for id_attractor in range(self.n_attractors):
                goals_latent_space_batch[primitive_type == i, :, id_attractor] = self.goals_latent_space[i][id_attractor]

        return goals_latent_space_batch

    def get_encoding_batch(self, primitive_type):
        """
        When multi-model learning, encodes primitive id into one-hot code
        """
        encoding_batch = torch.zeros(primitive_type.shape[0], self.n_primitives).cuda()
        for i in range(self.n_primitives):
            encoding_batch[primitive_type == i] = self.primitives_encodings[i]

        return encoding_batch

    def encoder(self, x_t, primitive_type):
        """
        Maps task space state to latent space state (psi)
        """
        # Get batch encodings
        if primitive_type.ndim == 1:  # if primitive type needs to be encoded
            encoding = self.get_encoding_batch(primitive_type)
        else:  # we assume the code is ready
            encoding = primitive_type

        # Encoder layer 1
        if self.multi_motion:
            input_encoded = torch.cat((x_t.cuda(), encoding), dim=1)
            e_1 = self.activation(self.norm_e_1(self.encoder1(input_encoded)))
        else:
            e_1 = self.activation(self.norm_e_1(self.encoder1(x_t.cuda())))

        # Encoder layer 2
        e_2 = self.activation(self.norm_e_2(self.encoder2(e_1)))

        # Encoder layer 3
        e_3 = self.activation(self.norm_e_3(self.encoder3(e_2)))

        e_4 = self.activation(self.encoder4(e_3))

        return e_4

    def decoder_dx(self, y_t):
        """
        Maps latent space state to task space derivative (phi)
        """
        # Normalize y_t
        y_t_norm = self.norm_de_dx0[0](y_t)

        # Decoder dx layer 1
        de_1 = self.activation(self.norm_de_dx1(self.decoder1_dx(y_t_norm)))

        # Decoder dx layer 2
        de_2 = self.activation(self.norm_de_dx2(self.decoder2_dx(de_1)))

        # Decoder dx layer 3
        de_3 = self.activation(self.norm_de_dx3(self.decoder3_dx(de_1)))
        de_4 = self.decoder4_dx(de_3)
        return de_4

    def gains_latent_dynamical_system(self, y_t_norm):
        """
        Computes gains latent dynamical system f^{L}
        """
        if self.adaptive_gains:
            input = y_t_norm
            latent_gain_1 = self.activation(self.norm_gain_1(self.gain_nn_1(input)))
            gains = self.sigmoid(self.gain_nn_2(latent_gain_1))

            # Keep gains between the set limits
            gains = gains * (self.latent_gain_upper_limit - self.latent_gain_lower_limit) + self.latent_gain_lower_limit
        else:
            gains = self.latent_gain
        return gains

    def latent_gain_limit_cycle(self, y_t_norm):
        """
        Computes gains latent dynamical system f^{L} for limit cycle component
        """

        if self.adaptive_gains:
            input = y_t_norm
            latent_gain_1 = self.activation(self.norm_gain_beta(self.gain_nn_beta_1(input)))
            gains = self.sigmoid(self.gain_nn_beta_2(latent_gain_1))

            # Keep gains between the set limits
            gains = gains * (self.latent_gain_upper_limit_beta - self.latent_gain_lower_limit_beta) + self.latent_gain_lower_limit
        else:
            gains = self.latent_gain
        return gains

    def latent_dynamical_system(self, y_t, primitive_type):
        """
        Stable latent dynamical system
        """
        if primitive_type.ndim > 1:  # if primitive is already encoded, decode TODO: this should be modified to work with changing goal position
            primitive_type = torch.argmax(primitive_type, dim=1)  # one hot encoding to integers

        # Get latent goals batch
        y_goals = self.get_goals_latent_space_batch(primitive_type)

        # With bad hyperparams y value can explode when simulating the system, creating nans -> clamp to avoid issues when hyperparam tuning
        y_t = torch.clamp(y_t, min=-3e18, max=3e18)

        # Normalize y_t
        y_t_norm = self.norm_latent_gain_input(y_t)

        # Get gain latent dynamical system
        alpha = self.gains_latent_dynamical_system(y_t_norm)

        # First order dynamical system in latent space
        if self.latent_system_dynamic_type == 'standard':
            dy_t = self.standard_latent_system(alpha, y_t, y_goals)

        elif self.latent_system_dynamic_type == "multivariate":
            dy_t = alpha * self.multivariate_pot_vectorized(y_t, y_goals)

        elif self.latent_system_dynamic_type == "multivariate_s2":
            dy_t = alpha * self.hyperspherical_flow(y_t, y_goals)

        elif self.latent_system_dynamic_type == "torus":
            dy_t = -alpha * self.hyperTorus_flow(y_t, y_goals)

        elif self.latent_system_dynamic_type == "gaussian":
            dy_t = alpha * self.gaussian_latent_system(y_t, y_goals)

        # the following dynamics are used for generate a continuous curve of attraction,
        # the difference between the 2 dynamic are in the single attractor component.
        # In one case the goal of that part of the dynamic is the 0 vector --> "limit cycle"
        # In the other the goal is the average point of the task space shape ---> "limit cycle avg"
        # having as goal the 0 vector generate a flow in the direction of the curve also from the internal area,
        # while having as goal the avg point generate a flat vector field inside the shape
        # ---------------------------------------------------------------------------------------
        elif self.latent_system_dynamic_type == "limit cycle":
            beta = self.latent_gain_limit_cycle(y_t_norm)
            dy_t = self.limit_cycle(y_t, alpha, beta, linear_dyn_goal=False)
        elif self.latent_system_dynamic_type == "limit cycle avg":
            beta = self.latent_gain_limit_cycle(y_t_norm)
            dy_t = self.limit_cycle(y_t, alpha, beta, linear_dyn_goal=True)
        else:
            raise NameError(f"{self.latent_system_dynamic_type} is not valid")
        return dy_t

    def limit_cycle(self, y_t, alpha, beta, linear_dyn_goal):
        y_t_polar = polar.euclidean_to_polar(y_t)
        y_t_polar_dot = self.polar_dyn(y_t_polar, beta, linear_dyn_goal)
        y_t_euc_dot = polar.polar_to_euclidean_velocity(y_t_polar, y_t_polar_dot)
        return alpha * y_t_euc_dot

    def polar_dyn(self, y_t_polar, beta=3.5, linear_dyn_goal=False):

        y_dot_pol = torch.zeros_like(y_t_polar)
        # logistic map
        y_dot_pol[:, 0] = y_t_polar[:, 0] * (1 - y_t_polar[:, 0]**2) * beta.T

        # zero angular velocity --> y_dot_pol[:, 2] = 0

        # linear goal of the dynamic -> 0 or avg y_g?
        if linear_dyn_goal:
            y_dot_pol[:, 2:] = -(self.y_g_avg[2:] - y_t_polar[:, 2:])**2
        else:
            y_dot_pol[:, 2:] = -y_t_polar[:, 2:]

        return y_dot_pol


    def hyperTorus_flow(self, y_t, y_g):

        """
        x: [B, D]
        Returns: dx/dt [B, D]
        """
        self.D = y_g.shape[1]  # D should be 300 based on your error

        y_t_exp = y_t[:, None, :]                 # [B, 1, D]
        goals_exp = y_g[0, :, :].unsqueeze(0)               # [1, n_goals, D]

        # Create pi as a scalar on the correct device
        pi = torch.tensor(torch.pi, device=y_t.device)

        # Delta calculation with proper broadcasting
        delta = (y_t_exp - goals_exp.permute(0, 2, 1) + pi) % (2 * pi) - pi  # [B, n_goals, D]

        # Rest of your code remains the same
        dist_sq = (2 * torch.sin(delta / 2)).pow(2).mean(dim=-1)  # [B, n_goals]
        weights = torch.softmax(-dist_sq / (2 * self.sigma**2), dim=1)  # [B, n_goals]
        dxdt = (weights[:, :, None] * delta).sum(dim=1)  # [B, D]

        return dxdt




    def hyperspherical_flow_test(self, y_t, y_g):
        y_g = y_g.permute(0, 2, 1)
        y_t = y_t.unsqueeze(1)
        diff = y_g - y_t
        square_diff = diff**2
        exp_arg = -torch.sum((square_diff/self.sigma**2), dim=2)
        exp_vals = torch.exp(exp_arg)
        potential = torch.sum(exp_vals.unsqueeze(-1) * diff * 2, dim=1)

        # projection to the tangent space:
        # ----------------
        # batched dot product:
        dot_weight = torch.einsum('bi, bi -> b', potential, y_t.squeeze(1))
        dot_weight = dot_weight.unsqueeze(1)  # [280 - 280, 1]potential -= torch.dot(potential, y_t) * y_t
        return potential - (dot_weight * y_t.squeeze(1))

    def hyperspherical_flow(self, y_t, y_g):
        """
        Compute hyperspherical vector field flow.

        Args:
            y_t: Tensor of shape [batch, state_dim] - current states
            y_g: Tensor of shape [state_dim, N_goal] - goal states
            beta: float - scaling factor for similarity
            eps: float - small number for numerical stability

        Returns:
            flow: Tensor of shape [batch, state_dim]
        """
        # Normalize
        y_g = y_g[0, ...]  # remove first dimension ---> was used for multi-motion learning
        y_t = torch.nn.functional.normalize(y_t, dim=1)  # [batch, Dim]
        y_g = torch.nn.functional.normalize(y_g, dim=0)  # [Dim, N_attractor]

        # ------------ Greater circle distance
        # Cosine similarities: [batch, N]
        cos_sim = torch.matmul(y_t, y_g)  # y_t: [B, D] x y_g: [D, N] -> [B, N]
        cos_sim = torch.clamp(cos_sim, -0.9999, 0.9999)  # for arccos stability

        # Angles: [batch, N]
        theta = torch.acos(cos_sim)  # [B, N]  ---> theta is the greater circle distance

        weights = torch.exp(-(theta**2) / self.sigma**2).unsqueeze(-1)

        # scaling factor --> what if theta is 0 ?
        # consider a sin expansion for replace that values
        sin_theta = torch.sin(theta)  # [B, N]

        # Term: y_g - cos(theta) * y_t for each goal
        # [B, N, D] = [B, N, 1] * [B, 1, D]
        cos_theta_y_t = cos_sim.unsqueeze(-1) * y_t.unsqueeze(1)  # [B, N, D]
        y_g_exp = y_g.T.unsqueeze(0).expand(y_t.shape[0], -1, -1)  # [B, N, D]

        log_map = y_g_exp - cos_theta_y_t  # [B, N, D]
        scale = (theta / sin_theta).unsqueeze(-1)  # [B, N, 1]
        tangent_vecs = scale * log_map  # [B, N, D]

        # Weighted sum
        flow = (weights * tangent_vecs).sum(dim=1)  # [B, D]

        return flow
    def torus_flow(self, y_t, y_g, sigma=0.7, min_speed=1e-3):
        """
        Compute the tangent-space dynamics on an N-dimensional torus.

        Args:
            y_t: Tensor of shape [B, D]         - current batch of torus points
            y_g: Tensor of shape [B, D, N]      - goal attractors per batch element
            sigma: float                        - Gaussian field width
            min_speed: float                    - minimum flow speed to avoid stagnation

        Returns:
            flow: Tensor of shape [B, D]        - tangent vector field on the torus
        """
        # --- Log map: minimal angular difference (modulo 2π) ---
        # [B, D, N] = ([B, D, N] - [B, D, 1]) % 2π
        diff = (y_g - y_t.unsqueeze(-1) + torch.pi) % (2 * torch.pi) - torch.pi  # [B, D, N]

        # --- Norm of log vectors ---
        dist = torch.norm(diff, dim=1)  # [B, N]

        # --- Safe direction normalization ---
        eps = 1e-8
        direction = diff / (dist.unsqueeze(1) + eps)  # [B, D, N]

        # --- Gaussian weighting ---
        weight = torch.exp(-(dist**2) / sigma**2)  # [B, N]
        speed = (2 / sigma**2) * weight + min_speed  # [B, N]

        # --- Compute final flow as weighted sum of directions ---
        flow = (speed.unsqueeze(1) * direction).sum(dim=-1)  # [B, D]

        return flow
    def torus_log_map(self, y_t, y_g):
        """
        Torus log map for batched data.

        Args:
            y_t: Tensor [B, D]           - current states (B=batch, D=dim)
            y_g: Tensor [B, D, N_goal]   - goal states (per-sample, per-dim)

        Returns:
            log_map: Tensor [B, D, N_goal] - tangent vectors
        """
        # Ensure shapes are compatible
        # y_t: [B, D] -> [B, D, 1] to broadcast
        diff = y_g - y_t.unsqueeze(-1)  # [B, D, N]

        # Wrap into [-pi, pi)
        log_map = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        return log_map

    def standard_latent_system(self, alpha, y_t, y_goals):
        """ single attractor lin dyn alpha*(y_g - y) """
        # use just the first attractors -> the only one if we have a single attractor point
        return alpha*(y_goals[:, :, 0].cuda() - y_t.cuda())

    def multivariate_pot(self, y_t, goal_list):
        """
        Multivariate gaussian dynamic on R^n. This version is not used, but keep it as reference for the first implementation
        pls use the function "multivariate_pot_vectorized"
        """
        goal_list = [goal_list[:, i] for i in range(goal_list.shape[1])]
        potential = torch.zeros(y_t.shape).cuda()   # batch x state dim
        for d in range(y_t.shape[1]):   # iterate over dimension
            for g in goal_list:
                exp_arg = -torch.sum((g - y_t)**2, dim=1)   # shape: batch
                potential[:, d] += torch.exp(exp_arg) * 2 * (g[d] - y_t[:, d])
        return potential

    def multivariate_pot_vectorized(self, y_t, y_g):
        """
        Multivariate gaussian dynamic on R^n
        """

        y_g = y_g.permute(0, 2, 1)
        y_t = y_t.unsqueeze(1)
        diff = y_g - y_t
        square_diff = diff**2
        exp_arg = -torch.sum(square_diff, dim=2)
        exp_vals = torch.exp(exp_arg)
        potential = torch.sum(exp_vals.unsqueeze(-1) * diff * 2, dim=1)
        return potential

    def gaussian_latent_system(self, y_t, y_g):
        """ use as latent state the sum of univariate gaussian """
        dy_t = y_g.cuda() - y_t.unsqueeze(-1).cuda()
        f_dot = -torch.exp(-dy_t ** 2) * (-2 * dy_t)
        f_dot = torch.sum(f_dot, axis=-1)
        return f_dot

    def well_known_2_point(self, alpha, y_t, y_goals):
        pass


