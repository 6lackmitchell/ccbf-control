from core.simulate import simulate

vehicle = "nonlinear_1d"
level = "default"
situation = "academic_example"

end_time = 4.0
timestep = 0.01

success = simulate(end_time, timestep, vehicle, level, situation)
