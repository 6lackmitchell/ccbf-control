from simulate import simulate

vehicle = 'bicycle'
level = 'dynamic'
situation = 'intersection'

if __name__ == "__main__":
    end_time = 10.0
    timestep = 0.01

    simulate(end_time, timestep, vehicle, level, situation)
