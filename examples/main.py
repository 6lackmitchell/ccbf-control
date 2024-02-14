from core.simulate import simulate

vehicle = "bicycle"
level = "dynamic"
situation = "swarm"

end_time = 50.0
timestep = 0.02

success = simulate(end_time, timestep, vehicle, level, situation)

import sys

sys.exit(int(success))
