from numpy import pi, array

# Acceleration due to gravity
G = 9.81  # meters / sec^2

# Vehicle Parameters
R = 1.0   # Safety radius (in m)
Lf = 1.0  # Front wheelbase (in m)
Lr = 1.0  # Rear wheelbase (in m)

# Control input constraints
ar_max = 0.3 * G
w_max = pi / 4
u_max = array([w_max, ar_max])

# Road Parameters
LW = 3.0  # Lane width (in m)