from core.simulate import simulate

vehicle = "double_integrator"
level = "dynamic"
situation = "toy_example"

end_time = 10.0
timestep = 1e-3

success = simulate(end_time, timestep, vehicle, level, situation)
