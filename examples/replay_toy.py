from sys import platform
from core.visualize import visualize

vehicle = "bicycle"
level = "dynamic"
situation = "toy_example"

end_time = 10.0
timestep = 0.01

root_dir = "/Documents/git/ccbf-control/data"

if platform == "linux" or platform == "linux2":
    # linux
    root_dir = "/home/6lackmitchell" + root_dir
elif platform == "darwin":
    # OS X
    root_dir = "/Users/mblack" + root_dir
elif platform == "win32":
    # Windows...
    pass

success = visualize(vehicle, level, situation, root_dir)
