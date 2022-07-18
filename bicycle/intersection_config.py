from agent import Agent
from .controllers.cbfs import *
from .controllers.ffcbf_qp_controller import compute_control as ffcbf_controller
from .controllers.nominal_controllers import u_nom_wrapper, intersection_controller_lqr
from .dynamics import *
from .timing import *

# Some nominal controllers need to be wrapped in u_nom_wrapper to pass the standardized arguments (t, x, id)
nominal_controller = intersection_controller_lqr

# Configure timing parameters
time = [dt, tf]

# Define Agents
agents = [Agent(0, z0[0, :], u0, cbf0, time, step_dynamics, ffcbf_controller, nominal_controller),
          Agent(1, z0[1, :], u0, cbf0, time, step_dynamics, ffcbf_controller, nominal_controller),
          Agent(2, z0[2, :], u0, cbf0, time, step_dynamics, ffcbf_controller, nominal_controller),
          Agent(3, z0[3, :], u0, cbf0, time, step_dynamics, nominal_controller),
         ]

# Get Filepath
delimiter = '/'  # linux or darwin or linux2
if platform == "win32":
    # Windows
    delimiter = '\\'

folder_ending = delimiter + 'datastore' + \
                delimiter + 'intersection' + \
                delimiter + 'accel_control' + \
                delimiter
save_path = os.path.dirname(os.path.abspath(__file__)) + folder_ending
