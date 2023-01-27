# Determine which problem is to be simulated
import builtins
import importlib


def visualize(vehicle: str, level: str, situation: str, root_dir: str) -> bool:
    """Plots simulation results for the system specified by the arguments.

    Arguments:
        vehicle: the vehicle to be simulated
        level: the control level (i.e. kinematic, dynamic, etc.)
        situation: i.e. intersection_old, intersection, etc.

    Returns:
        success: true / false flag

    """
    filepath = root_dir + f"/{vehicle}/{level}/{situation}/"

    # Make problem config available to other modules
    builtins.PROBLEM_CONFIG = {
        "vehicle": vehicle,
        "control_level": level,
        "situation": situation,
        "system_model": "deterministic",
    }
    mod = "models.{}.{}.{}.vis_testing".format(vehicle, level, situation)
    # mod = "models.{}.{}.{}.vis_paper".format(vehicle, level, situation)

    # Problem-specific import
    module = importlib.import_module(mod)
    # globals().update({"replay": getattr(module, "replay")})

    # filename = None  # "baseline_no_estimation.pkl"
    # return replay(filepath, filename)
