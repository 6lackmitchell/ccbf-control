import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator as rgi

def derivative(f,
               a,
               method='central',
               h=1e-2):
    """Compute the difference formula for f'(a) with step size h.

    Borrowed from UBC Math Deptartment website.

    INPUTS
    ------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    OUTPUTS
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h
    """
    if method == 'central':
        return (f(a + h) - f(a - h))/(2*h)
    elif method == 'forward':
        return (f(a + h) - f(a))/h
    elif method == 'backward':
        return (f(a) - f(a - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")

def partial_derivative(f,
                       a,
                       direction='x',
                       h=1e-2):
    """Compute the difference formula for f'(a) with step size h.

    Borrowed from UBC Math Deptartment website.

    INPUTS
    ------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    OUTPUTS
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h
    """
    if direction == 'x':
        hnew = np.array([h, 0.0, 0.0])
    elif direction == 'y':
        hnew = np.array([0.0, h, 0.0])
    elif direction == 'z':
        hnew = np.array([0.0, 0.0, h])
    else:
        raise ValueError("Direction must be 'x', 'y' or 'z'.")
    return (f(a + hnew) - f(a - hnew))/(2*h)

# Load Wind Data
wind_file = loadmat('/home/mblack/Documents/data/wind_velocity_data.mat')
wind_u    = wind_file['u2']
wind_v    = wind_file['v2']
wind_w    = wind_file['w2']
# wind_u = np.ones((1000,1000,1000))
# wind_v = np.ones((1000,1000,1000))
# wind_w = np.ones((1000,1000,1000))

# Scale the winds
SCALE = 1.#00.

# Configure wind mesh
xlim      = 5.0
ylim      = 5.0
zlim      = 5.0
xx        = np.linspace(-xlim,xlim,wind_u.shape[0])
yy        = np.linspace(-ylim,ylim,wind_v.shape[1])
zz        = np.linspace(-0.2,zlim,wind_w.shape[2])

# Configure Wind Interpolating Functions
windu_interp = rgi((xx,yy,zz),wind_u / SCALE)
windv_interp = rgi((xx,yy,zz),wind_v / SCALE)
windw_interp = rgi((xx,yy,zz),wind_w / SCALE)

if __name__ == "__main__":
    print(np.max(abs(wind_u)))
    print(np.max(abs(wind_v)))
    print(np.max(abs(wind_w)))