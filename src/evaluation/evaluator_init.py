from evaluation.evaluate_2d_o1 import Evaluate2DO1
from evaluation.evaluate_2d_o2 import Evaluate2DO2
from evaluation.evaluate_3d import Evaluate3D
from evaluation.evaluate_nd import EvaluateND
from evaluation.evaluate_shape import EvaluateShape
from evaluation.evaluate_7d_o1 import Evaluate7DR3S3


def evaluator_init(learner, data, params, verbose=True):
    """
    Selects and initializes evaluation class
    """
    #if params.component_wise_plot:
    #    return EvaluateND(learner, data, params, verbose)

    if params.workspace_dimensions == 2 and params.dynamical_system_order == 1:
        if params.latent_dynamic_system_type == "limit cycle":
            return EvaluateShape(learner, data, params, verbose)
        return Evaluate2DO1(learner, data, params, verbose)
    elif params.workspace_dimensions == 2 and params.dynamical_system_order == 2:
        return Evaluate2DO2(learner, data, params, verbose)
    elif params.workspace_dimensions == 3:
        return Evaluate3D(learner, data, params, verbose)
    elif params.workspace_dimensions > 5:
        return Evaluate7DR3S3(learner, data, params, verbose)
    else:
        return EvaluateND(learner, data, params, verbose)
