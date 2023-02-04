from core.simulate import simulate

vehicle = "nonlinear_1d"
level = "default"
situation = "academic_example"

end_time = 10.0
timestep = 1e-3

success = simulate(end_time, timestep, vehicle, level, situation)
