# from .init_warehouse import *

import builtins
from importlib import import_module

vehicle = builtins.PROBLEM_CONFIG["vehicle"]
control_level = builtins.PROBLEM_CONFIG["control_level"]
situation = builtins.PROBLEM_CONFIG["situation"]
mod = "models." + vehicle + "." + control_level + "." + situation

# Programmatic version of 'from control_level import *'
try:
    module = import_module(mod)
    globals().update(
        {n: getattr(module, n) for n in module.__all__}
        if hasattr(module, "__all__")
        else {k: v for (k, v) in module.__dict__.items() if not k.startswith("_")}
    )
except ModuleNotFoundError as e:
    print("No module named '{}' -- exiting.".format(mod))
    raise e
