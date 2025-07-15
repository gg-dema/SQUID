import importlib

import sys
sys.path.append("./")

from simple_parsing import ArgumentParser
from initializer import initialize_framework


# Get arguments
parser = ArgumentParser()
parser.add_argument('--params', type=str, default='Discrete_Squid_S_1o_kuka', help='')
parser.add_argument('--results-base-directory', type=str, default='./', help='')

args = parser.parse_args()

# Import parameters
Params = getattr(importlib.import_module('params.' + args.params), 'Params')
params = Params(args.results_base_directory)

params.results_path += params.selected_primitives_ids + '/'

params.load_model = True

# Initialize training objects
learner, evaluator, _ = initialize_framework(params, args.params)
metrics_acc, metrics_stab = evaluator.run(iteration=0)

