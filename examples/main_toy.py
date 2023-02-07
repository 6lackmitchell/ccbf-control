from core.simulate import simulate

vehicle = "bicycle"
level = "dynamic"
situation = "toy_example"

end_time = 10.0
timestep = 1e-3

success = simulate(end_time, timestep, vehicle, level, situation)
