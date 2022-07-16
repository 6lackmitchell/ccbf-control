# Determine which problem is to be simulated
import sys
import importlib
args = sys.argv
if len(args) > 1:
    config = str(args[1])
    if len(args) > 2:
        import builtins
        builtins.VIS_FILE = str(args[2])
else:
    config = 'bicycle'

# Problem-specific import
try:
    # module = importlib.import_module(config + '.vis')
    module = importlib.import_module(config + '.vis_mc')
except ModuleNotFoundError:
    # print('No module named \'{}\' -- exiting.'.format(config + '.vis'))
    print('No module named \'{}\' -- exiting.'.format(config + '.vis_mc'))
    sys.exit()

# if config == 'single_integrator':
#     # import single_integrator.vis
#     import single_integrator.vis_mc
#
# if config == 'simple2ndorder':
#     import viz.simple2ndorder_vis
# elif config == 'overtake':
#     import viz.overtake_vis
# elif config == 'quadrotor':
#     import viz.quadrotor_vis
# elif config == 'bicycle':
#     import viz.bicycle_vis_intersection
#     # import viz.bicycle_vis_highway
# elif config == 'bicycle_jerk':
#     import viz.bicycle_vis_intersection_jerk
#     # import viz.bicycle_vis_highway_jerk
# elif config == 'bicycle_highway_merging':
#     import viz.bicycle_vis_highway_merging