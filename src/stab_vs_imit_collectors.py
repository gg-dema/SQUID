import importlib

import torch
from simple_parsing import ArgumentParser
from initializer import initialize_framework
from torch.utils.tensorboard import SummaryWriter

import torch
import numpy as np

parameters_file = "double_multivariate_goal_mapping_099_lambda"
results_base_directory = "../../imit_vs_stab/"

# Import parameters
Params = getattr(importlib.import_module('params.' + parameters_file), 'Params')
params = Params(results_base_directory)
# 2 attractors
dataset_id = '0'
seed = 100
torch.manual_seed(seed)
np.random.seed(seed)
alpha_vector = np.linspace(1.5, 2, 15)

for alpha in alpha_vector:


    params = Params(results_base_directory)
    params.results_path += dataset_id + '/' + "{:.4f}".format(alpha) + '/'
    params.imitation_loss_weight = alpha
    params.selected_primitives_ids = dataset_id
    params.goal_mapping_loss_weight = 0.05
    params.stabilization_loss_weight = (2-alpha)
    params.cycle_loss_weight = 0.0
    params.fixed_point_iteration_thr = 3
    params.max_iterations = 10_000
    params.dataset_name = "eval_dataset"

    # Initialize training objects
    learner, evaluator, _ = initialize_framework(params, parameters_file)

    # Train
    for iteration in range(params.max_iterations + 1):
        # Evaluate model
        if iteration % params.evaluation_interval == 0:
            metrics_acc, metrics_stab = evaluator.run(iteration=iteration)
            if params.save_evaluation:
                evaluator.save_progress(params.results_path, iteration, learner.model, None)

            print('Metrics sum:', metrics_acc['metrics sum'], '; Number of unsuccessful trajectories:', metrics_stab['n spurious'])

        # Training step
        loss, loss_list, losses_names = learner.train_step()

        # Print progress
        if iteration % 10 == 0:
            print_msg = ' '
            for j in range(len(losses_names)):
                print_msg += " " + str(losses_names[j]) + " " + str(loss_list[j].item())
            print(iteration, 'Total cost:', loss.item(), print_msg)


