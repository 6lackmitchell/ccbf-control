from simulate import simulate

vehicle = 'bicycle'
level = 'dynamic'
situation = 'intersection'

if __name__ == "__main__":
    end_time = 50.0
    timestep = 0.05

    success = simulate(end_time, timestep, vehicle, level, situation)
