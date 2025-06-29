import importlib

import torch
from simple_parsing import ArgumentParser
from initializer import initialize_framework
from torch.utils.tensorboard import SummaryWriter

# seed if needed
from torch import manual_seed
from numpy.random import seed
seed(3)
manual_seed(3)

# Get arguments
parser = ArgumentParser()
parser.add_argument('--params', type=str, default='Discrete_Squid_R2_1o', help='')
parser.add_argument('--results-base-directory', type=str, default='./', help='')

args = parser.parse_args()

# Import parameters
Params = getattr(importlib.import_module('params.' + args.params), 'Params')
params = Params(args.results_base_directory)

params.results_path += params.selected_primitives_ids + '/'

# Initialize training objects
learner, evaluator, _ = initialize_framework(params, args.params)

# Start tensorboard writer
log_name = args.params + '_' + params.selected_primitives_ids
writer = SummaryWriter(log_dir='results/tensorboard_runs/' + log_name)

# Train
for iteration in range(params.max_iterations + 1):
    # Evaluate model
    if iteration % params.evaluation_interval == 0:
        metrics_acc, metrics_stab = evaluator.run(iteration=iteration)
        if params.save_evaluation:
            evaluator.save_progress(params.results_path, iteration, learner.model, writer)

        if params.latent_dynamic_system_type in ("limit cycle", "limit cycle avg") and params.workspace_dimensions == 2:
            print("Metrics chamfer", metrics_stab["chamfer"])
        else:
            print('Metrics sum:', metrics_acc['metrics sum'], '; Number of unsuccessful trajectories:', metrics_stab['n spurious'])

    # Training step
    loss, loss_list, losses_names = learner.train_step()

    # Print progress
    if iteration % 10 == 0:
        print_msg = ' '
        for j in range(len(losses_names)):
            print_msg += " " + str(losses_names[j]) + " " + str(loss_list[j].item())
        print(iteration, 'Total cost:', loss.item(), print_msg)

    # Log losses in tensorboard
    for j in range(len(losses_names)):
        writer.add_scalar('losses/' + losses_names[j], loss_list[j], iteration)

