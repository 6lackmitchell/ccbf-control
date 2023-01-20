from core.simulate import simulate

vehicle = "double_integrator"
level = "dynamic"
situation = "toy_example"

end_time = 20.0
timestep = 0.01

success = simulate(end_time, timestep, vehicle, level, situation)
