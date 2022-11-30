from simulate import simulate

vehicle = "bicycle"
level = "dynamic"
situation = "warehouse"

end_time = 40.0
timestep = 0.001

success = simulate(end_time, timestep, vehicle, level, situation)
