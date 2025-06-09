from dataclasses import dataclass
import numpy as np

@dataclass
class Params:
    """ General parameters """
    dataset_name: str = 'joint_space'  # selects dataset, (multi-attractors, kuka, joint_space)
    results_path: str = 'results/post_laurea/discrete_multi_attractor_joint_space/'
    multi_motion: bool = False  # true when learning multiple motions together
    selected_primitives_ids: str = '3'  # id number from dataset_keys.py, e.g., '2' or '4,0,6'
    workspace_dimensions: int = 7  # dimensionality of the data
    spherical_latent_space: bool = False
    saturate_out_of_boundaries_transitions: bool = True  # True to enforce positively invariant set
    dynamical_system_order: int = 1  # options: 1, 2

    """ Multi Attractors """
    n_attractors: int = 2
    latent_dynamic_system_type: str = "multivariate"  # option: standard, gaussian, well_known_gaussian, well_known_multivariate, multivariate
    shaped_attractors_form: str = None

    """ Latent Dynamical System parameters """
    adaptive_gains: bool = True  # adaptive gains if true
    latent_gain_lower_limit: float = 0  # adaptive gains lower limit (always zero in paper)
    latent_gain_upper_limit: float = 0.0997  # adaptive gains upper limit
    latent_gain: float = 0.008  # value of gains when fixed

    """ Neural Network """
    latent_space_dim: int = 300  # dimensionality latent space
    neurons_hidden_layers: int = 400  # number of neurons per layer
    batch_size: int = 126  # sampling batch size
    batch_size_contrastive_norm: int = 0
    batch_size_orientation_loss: int = 0
    learning_rate: float = 0.0000039  # AdamW learning rate
    weight_decay: float = 0.00000001  # AdamW weight decay

    """ Contrastive Imitation """
    imitation_loss_weight: float = 1.0  # imitation loss weight
    stabilization_loss_weight: float = 1  # stability loss weight
    goal_mapping_loss_weight: float = 1
    goal_mapping_decreasing_weight: float = 0.997
    boundary_loss_weight: float = 0.05
    cycle_loss_weight: float = 0.0
    cycle_orientation_loss_weight: float = 0.0
    imitation_window_size: int = 5  # 15 original  # imitation window size
    stabilization_window_size: int = 3  # 2 original  # stability window size
    stabilization_loss: str = 'contrastive'  # options: contrastive, triplet
    contrastive_margin: float = 0.00333  # contrastive loss margin
    contrastive_norm_margin: float = 0.0
    triplet_margin: float = 1e-4  # triplet loss margin
    interpolation_sigma: float = 0.8  # percentage of points sampled in demonstrations space when multi-model learning

    """ Training """
    train: bool = True  # true when training
    load_model: bool = False  # true to load previously trained model
    max_iterations: int = 41000  # maximum number of training iterations

    """ Preprocessing """
    spline_sample_type: str = 'from data'  # resample from spline type, options: from data, from data resample, evenly spaced
    workspace_boundaries_type: str = 'from data'  # options: from data, custom
    workspace_boundaries: str = np.array(
    [
        [-3.14, 3.14],
        [-3.14, 3.14],
        [-3.14, 3.14],
        [-3.14, 3.14],
        [-3.14, 3.14],
        [-3.14, 3.14],
        [-3.14, 3.14]
    ])  # list to provide boundaries when workspace_boundaries_type = custom
    trajectories_resample_length: int = 2000  # amount of points resampled from splines when type spline_sample_type is 'from data resample' or 'evenly spaced'
    state_increment: float = 0.6  # when workspace_boundaries_type = from data, percentage to increment state-space size

    """ Evaluation """
    save_evaluation: bool = True  # true to save evaluation results
    evaluation_interval: int = 1000  # interval between training iterations to evaluate model
    quanti_eval: bool = True  # quantitative evaluation
    quali_eval: bool = True  # qualitative evaluation
    diffeo_quanti_eval: bool = True  # quantitative evaluation of diffeomorphism mismatch
    diffeo_quali_eval: bool = True  # qualitative evaluation of diffeomorphism mismatch
    ignore_n_spurious: bool = False  # when selecting best model, true to ignore amount of spurious attractors
    fixed_point_iteration_thr = 3  # distance threshold to consider that a point did not reach the goal
    density: int = 2  # density^workspace_dimension = amount of points sampled from state space for evaluation
    diffeo_comparison_subsample: int = 7  # state space points subsample for qualitative diffeomorphism comparison
    simulated_trajectory_length: int = 6000  # integration length for evaluation
    diffeo_comparison_length: int = 50  # integration length of diffeomorphism comparison
    evaluation_samples_length: int = 100  # integration steps skipped in quantitative evaluation for faster evaluation
    show_plotly = True

    """ Hyperparameter Optimization """
    gamma_objective_1 = 0.48  # weight 1 for hyperparameter evaluation
    gamma_objective_2 = 3.5  # weight 2 for hyperparameter evaluation
    optuna_n_trials = 1000  # maximum number of optuna trials

    """ Dataset training """
    length_dataset = 30  # number of primitives in dataset

    def __init__(self, results_base_directory):
        self.results_path = results_base_directory + self.results_path
