import importlib

import torch
from simple_parsing import ArgumentParser
from initializer import initialize_framework
from torch.utils.tensorboard import SummaryWriter

import torch
import numpy as np

from torch import manual_seed
from numpy.random import seed

parameters_file = "double_multivariate_goal_mapping_099_lambda"
results_base_directory = "../../results_eval_collector/condor_no_imit/"

# Import parameters
Params = getattr(importlib.import_module('params.' + parameters_file), 'Params')
params = Params(results_base_directory)
# 2 attractors
# dataset_ids = []
# 3 attractors
dataset_ids = ['0', '1', '2', '3', '4', '5', '6']

for dataset_id in dataset_ids:
    seed_list = [100, 101, 102, 103]
    for seed in seed_list:

        torch.manual_seed(seed)
        np.random.seed(seed)
        params = Params(results_base_directory)
        print(dataset_id, params.results_path)
        params.selected_primitives_ids = dataset_id
        params.results_path += dataset_id + f"/seed_{seed}/"

        if int(dataset_id) < 3:
            params.n_attractors = 2
        else:
            params.n_attractors = 3

        params.imitation_loss_weight = 0.0
        params.goal_mapping_loss_weight = 0.05
        params.stabilization_loss_weight = 0.5
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


