from evaluation.evaluate_2d_o1 import Evaluate2DO1
from evaluation.evaluate_2d_o2 import Evaluate2DO2
from evaluation.evaluate_3d import Evaluate3D
from evaluation.evaluate_nd import EvaluateND
from evaluation.evaluate_shape import EvaluateShape
from evaluation.evaluate_3d_shape import Evaluate3D_shape
from evaluation.evaluate_7d_o1 import Evaluate7DR3S3
from evaluation.evaluate_7D_frame import Evaluate7D_frame
from evaluation.evalute_joint_space import EvaluateCircularND

def evaluator_init(learner, data, params, verbose=True):
    """
    Selects and initializes evaluation class
    """
    #if params.component_wise_plot:
    #    return EvaluateND(learner, data, params, verbose)
    if params.latent_dynamic_system_type in ("limit cycle", "limit cycle avg"):
        if params.workspace_dimensions == 3:
            return Evaluate3D_shape(learner, data, params, verbose)
        if params.workspace_dimensions == 2:
            return EvaluateShape(learner, data, params, verbose)

    if params.workspace_dimensions == 2 and params.dynamical_system_order == 1:
        return Evaluate2DO1(learner, data, params, verbose)
    elif params.workspace_dimensions == 2 and params.dynamical_system_order == 2:
        return Evaluate2DO2(learner, data, params, verbose)
    elif params.workspace_dimensions == 3:
        return Evaluate3D(learner, data, params, verbose)
    elif params.workspace_dimensions == 7:
        if params.joint_space:
            return EvaluateCircularND(learner, data, params, verbose)
        elif params.pose_tracking == "SE3":
            return Evaluate7D_frame(learner, data, params, verbose)
        elif params.pose_tracking == "Component-wise":
            return Evaluate7DR3S3(learner, data, params, verbose)
    else:
        return EvaluateND(learner, data, params, verbose)
