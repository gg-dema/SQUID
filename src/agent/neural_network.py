import torch
import numpy as np
import agent.utils.polar as polar
from agent.utils.ranking_losses import great_circle_distance as gcd_ranking

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
        elif self.latent_system_dynamic_type in ("multivariate", "multivariate_s2"):            # multiple gain for each equation of the latent state ---> unbalance behaviour, keep for now
            # a single alpha parameters for  (if use it back, change the dim of alpha --> look at multivariate dynamics)
            # return torch.nn.Linear(self.latent_space_dim, latent_input_size*self.n_attractors)
            return torch.nn.Linear(self.latent_space_dim, latent_input_size)
        elif self.latent_system_dynamic_type == "well known":
            raise NotImplementedError(' well known still to define ==> gain nn for alpha')
        elif self.latent_system_dynamic_type == 'limit cycle':
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
            the, compute the part used in the linear dynamic (the avg goals shape)
        """
        primitive_type = torch.FloatTensor([0]).cuda()
        goals_shape = torch.FloatTensor(goals_shape).cuda()

        self.y_g0 = self.encoder(goals_shape, primitive_type).mean(0)


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

        elif self.latent_system_dynamic_type == "gaussian":
            dy_t = alpha * self.gaussian_latent_system(y_t, y_goals)

        elif self.latent_system_dynamic_type == "limit cycle":
            beta = self.latent_gain_limit_cycle(y_t_norm)
            dy_t = self.limit_cycle(y_t, y_goals, alpha, beta)
        else:
            raise NameError(f"{self.latent_system_dynamic_type} is not valid")
        return dy_t


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


    def limit_cycle(self, y_t, y_goals, alpha, beta):
        y_t_polar = polar.euclidean_to_polar(y_t)
        y_t_polar_dot = self.polar_dyn(y_t_polar, y_goals, beta)
        y_t_euc_dot = polar.polar_to_euclidean_velocity(y_t_polar, y_t_polar_dot)
        return alpha * y_t_euc_dot

    def polar_dyn(self, y_t_polar, y_goals, beta=3.5):

        y_goals_avg = torch.mean(y_goals, -1)
        y_dot_pol = torch.zeros_like(y_t_polar)
        y_dot_pol[:, 0] = y_t_polar[:, 0] * (1 - y_t_polar[:, 0]**2) * beta.T

        # y_dot_pol[:, 1] = 0 --> automatic done
        # then 3 option : this option generate different velocity profile
        # y_dot_pol[:, 2:] = torch.sum(y_goals[:, 2:, :] - y_t_polar[:, 2:].unsqueeze(-1), axis=-1)
        # y_dot_pol[:, 2:] = -(self.y_g0[2:] - y_t_polar[:, 2:])**2
        y_dot_pol[:, 2:] = -y_t_polar[:, 2:]
        return y_dot_pol

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
        y_g = y_g[0, ...]
        y_t = torch.nn.functional.normalize(y_t, dim=1)  # [batch, D]
        y_g = torch.nn.functional.normalize(y_g, dim=0)  # [D, N]

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

        # Weights
        # weights = torch.exp(-cos_sim).unsqueeze(-1)  # [B, N, 1]

        # Weighted sum
        flow = (weights * tangent_vecs).sum(dim=1)  # [B, D]

        return flow


    def standard_latent_system(self, alpha, y_t, y_goals):
        # use just the first attractors
        # maybe i should check the dim of the attractors
        return alpha*(y_goals[:, :, 0].cuda() - y_t.cuda())

    def multivariate_pot(self, y_t, goal_list):

        goal_list = [goal_list[:, i] for i in range(goal_list.shape[1])]
        potential = torch.zeros(y_t.shape).cuda()   # batch x state dim
        for d in range(y_t.shape[1]):   # iterate over dimension
            for g in goal_list:
                exp_arg = -torch.sum((g - y_t)**2, dim=1)   # shape: batch
                potential[:, d] += torch.exp(exp_arg) * 2 * (g[d] - y_t[:, d])
        return potential

    def multivariate_pot_vectorized(self, y_t, y_g):
        y_g = y_g.permute(0, 2, 1)
        y_t = y_t.unsqueeze(1)
        diff = y_g - y_t
        square_diff = diff**2
        exp_arg = -torch.sum(square_diff, dim=2)
        exp_vals = torch.exp(exp_arg)
        potential = torch.sum(exp_vals.unsqueeze(-1) * diff * 2, dim=1)
        return potential

    def gaussian_latent_system(self, y_t, y_g):
        dy_t = y_g.cuda() - y_t.unsqueeze(-1).cuda()
        f_dot = -torch.exp(-dy_t ** 2) * (-2 * dy_t)
        f_dot = torch.sum(f_dot, axis=-1)
        return f_dot

    def well_known_2_point(self, alpha, y_t, y_goals):
        pass

    def log_map_taylor_exp(x, y, eps=1e-8):
        # x: [batch, dim]
        # y: [batch, dim, n]
        dot = torch.clamp((x.unsqueeze(-1) * y).sum(dim=1), -1.0, 1.0)  # [batch, n]
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)

        # Compute scale factors using Taylor expansion for small angles
        scale = torch.where(
            theta < 1e-3,
            1 + theta**2 / 6 + 7 * theta**4 / 360,
            theta / torch.maximum(sin_theta, torch.tensor(eps, device=theta.device))
        )

        # Compute log map
        v = scale.unsqueeze(1) * (y - dot.unsqueeze(1) * x.unsqueeze(-1))
        return v
    def  great_circle_distance(a, b):
        # a: [batch, dim]
        # b: [batch, dim, n] or [batch, dim] (broadcastable)
        cosine = torch.clamp((a.unsqueeze(-1) * b).sum(dim=1), -0.99999, 0.99999)
        return torch.acos(cosine)
