"""__init__.py

Initializer for the cbfs module. Imports parameters and methods according to
the situation specified in the main executable.

"""
import builtins
from importlib import import_module

situation = builtins.PROBLEM_CONFIG["situation"]
mod = f"core.cbfs.__{situation.replace('_','')}__"

module = import_module(mod)
globals().update(
    {n: getattr(module, n) for n in module.__all__}
    if hasattr(module, "__all__")
    else {k: v for (k, v) in module.__dict__.items() if not k.startswith("_")}
)
