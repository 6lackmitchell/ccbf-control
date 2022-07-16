import numpy as np
# from .initial_conditions_intersection import z0

def cbf(z):
    return np.array([1.0])
    # return np.array([cbf_altitude(x),cbf_attitude(x)])
    # return np.array([cbf_altitude(x),cbf_lateral_outer(x),cbf_lateral_inner(x),cbf_velocity_outer(x),cbf_velocity_inner(x),cbf_attitude(x)])
    # return np.array([cbf_altitude(x),cbf_lateral(x),cbf_attitude(x)])

# ############################# Altitude Safety #############################
#
# def cbf_altitude(x):
#     return 1 - ((x[2] - CZ)/PZ)**NZ
#
# def cbfdot_altitude(x):
#     hdot_z = NZ * (x[2] - CZ)**(NZ-1) * zdot(x)
#     return -hdot_z / PZ**NZ
#
# def cbf2dot_altitude_uncontrolled(x):
#     h2dot_z = (NZ * (NZ-1) * (x[2] - CZ)**(NZ-2) * zdot(x)**2) + \
#               (NZ*(x[2] - CZ)**(NZ-1) * z2dot(x,1)[0])
#     return -h2dot_z / PZ**NZ
#
# def cbf2dot_altitude_controlled(x):
#     h2dot_z = NZ*(x[2] - CZ)**(NZ-1) * (z2dot(x,1)[1])
#     return -h2dot_z / PZ**NZ
#
# ############################# Lateral Safety #############################
#
# def cbf_lateral(x):
#     return 1 - ((x[0] - CX)/PX)**NX - ((x[1] - CY)/PY)**NY
#
# def cbfdot_lateral(x):
#     hdot_x = NX * (x[0] - CX)**(NX-1) / PX**NX * xdot(x)
#     hdot_y = NY * (x[1] - CY)**(NY-1) / PY**NY * ydot(x)
#     return -((hdot_x / PX**NX) + (hdot_y / PY**NY))
#
# def cbf2dot_lateral(x,F):
#     h2dot_x = (NX * (NX-1) * (x[0] - CX)**(NX-2) * xdot(x)**2) + \
#               (NX*(x[0] - CX)**(NX-1) * x2dot(x,F))
#     h2dot_y = (NY * (NY-1) * (x[1] - CY)**(NY-2) * ydot(x)**2) + \
#               (NY*(x[1] - CY)**(NY-1) * y2dot(x,F))
#     return -((h2dot_x / PX**NX) + (h2dot_y / PY**NY))
#
# def cbf3dot_lateral(x,F):
#     h3dot_x = NX*(NX-1)*(NX-2)*(x[0] - CX)**(NX-3)*xdot(x)**3 + \
#               NX*(NX-1)*(x[0] - CX)**(NX-2)*3*xdot(x)*x2dot(x,F) + \
#               NX*(x[0] - CX)**(NX-1)*x3dot(x,F)
#     h3dot_y = NY*(NY-1)*(NY-2)*(x[1] - CY)**(NY-3)*ydot(x)**3 + \
#               NY*(NY-1)*(x[1] - CY)**(NY-2)*3*ydot(x)*y2dot(x,F) + \
#               NY*(x[1] - CY)**(NY-1)*y3dot(x,F)
#     return -((h3dot_x / PX**NX) + (h3dot_y / PY**NY))
#
# def cbf4dot_lateral_uncontrolled(x,F):
#     h4dot_x = NX*(NX-1)*(NX-2)*(NX-3)*(x[0] - CX)**(NX-4)*xdot(x)**4 + \
#               NX*(NX-1)*(NX-2)*(x[0] - CX)**(NX-3)*5*xdot(x)**2*x2dot(x,F) + \
#               NX*(NX-1)*(x[0] - CX)**(NX-2)*(3*x2dot(x,F)**2 + 4*xdot(x)*x3dot(x,F)) + \
#               NX*(x[0] - CX)**(NX-1) * x4dot(x,F)[0]
#     h4dot_y = NY*(NY-1)*(NY-2)*(NY-3)*(x[1] - CY)**(NY-4)*ydot(x)**4 + \
#               NY*(NY-1)*(NY-2)*(x[1] - CY)**(NY-3)*5*ydot(x)**2*y2dot(x,F) + \
#               NY*(NY-1)*(x[1] - CY)**(NY-2)*(3*y2dot(x,F)**2 + 4*ydot(x)*y3dot(x,F)) + \
#               NY*(x[1] - CY)**(NY-1) * y4dot(x,F)[0]
#     return -((h4dot_x / PX**NX) + (h4dot_y / PY**NY))
#
# def cbf4dot_lateral_controlled(x,F):
#     h4dot_x = NX*(x[0] - CX)**(NX-1) * x4dot(x,F)[1]
#     h4dot_y = NY*(x[1] - CY)**(NY-1) * y4dot(x,F)[1]
#     return -((h4dot_x / PX**NX) + (h4dot_y / PY**NY))

# ########################## Inner/Outer Lateral Safety ##########################

# def cbf_lateral_outer(x):
#     return 1 - ((x[0] - CXo)/PXo)**NXo - ((x[1] - CYo)/PYo)**NYo

# def cbfdot_lateral_outer(x):
#     hdot_x = NXo * (x[0] - CXo)**(NXo-1) / PXo**NXo * xdot(x)
#     hdot_y = NYo * (x[1] - CYo)**(NYo-1) / PYo**NYo * ydot(x)
#     return -((hdot_x / PXo**NXo) + (hdot_y / PYo**NYo))

# def cbf2dot_lateral_outer(x,F):
#     h2dot_x = (NXo * (NXo-1) * (x[0] - CXo)**(NXo-2) * xdot(x)**2) + \
#               (NXo*(x[0] - CXo)**(NXo-1) * x2dot(x,F))
#     h2dot_y = (NYo * (NYo-1) * (x[1] - CYo)**(NYo-2) * ydot(x)**2) + \
#               (NYo*(x[1] - CYo)**(NYo-1) * y2dot(x,F))
#     return -((h2dot_x / PXo**NXo) + (h2dot_y / PYo**NYo))

# def cbf3dot_lateral_outer(x,F):
#     h3dot_x = NXo*(NXo-1)*3*xdot(x)*x2dot(x,F) + \
#               NXo*(x[0] - CXo)**(NXo-1)*x3dot(x,F)
#     h3dot_y = NYo*(NYo-1)*3*ydot(x)*y2dot(x,F) + \
#               NYo*(x[1] - CYo)**(NYo-1)*y3dot(x,F)
#     return -((h3dot_x / PXo**NXo) + (h3dot_y / PYo**NYo))

# def cbf4dot_lateral_outer_uncontrolled(x,F):
#     h4dot_x = NXo*(NXo-1)*(3*x2dot(x,F)**2 + 4*xdot(x)*x3dot(x,F)) + \
#               NXo*(x[0] - CXo)**(NXo-1) * x4dot(x,F)[0]
#     h4dot_y = NYo*(NYo-1)*(3*y2dot(x,F)**2 + 4*ydot(x)*y3dot(x,F)) + \
#               NYo*(x[1] - CYo)**(NYo-1) * y4dot(x,F)[0]
#     return -((h4dot_x / PXo**NXo) + (h4dot_y / PYo**NYo))

# def cbf4dot_lateral_outer_controlled(x,F):
#     h4dot_x = NXo*(x[0] - CXo)**(NXo-1) * x4dot(x,F)[1]
#     h4dot_y = NYo*(x[1] - CYo)**(NYo-1) * y4dot(x,F)[1]
#     return -((h4dot_x / PXo**NXo) + (h4dot_y / PYo**NYo))

# def cbf_lateral_inner(x):
#     return ((x[0] - CXi)/PXi)**NXi + ((x[1] - CYi)/PYi)**NYi - 1

# def cbfdot_lateral_inner(x):
#     hdot_x = NXi * (x[0] - CXi)**(NXi-1) / PXi**NXi * xdot(x)
#     hdot_y = NYi * (x[1] - CYi)**(NYi-1) / PYi**NYi * ydot(x)
#     return ((hdot_x / PXi**NXi) + (hdot_y / PYi**NYi))

# def cbf2dot_lateral_inner(x,F):
#     h2dot_x = (NXi * (NXi-1) * xdot(x)**2) + \
#               (NXi*(x[0] - CXi)**(NXi-1) * x2dot(x,F))
#     h2dot_y = (NYi * (NYi-1) * ydot(x)**2) + \
#               (NYi*(x[1] - CYi)**(NYi-1) * y2dot(x,F))
#     return ((h2dot_x / PXi**NXi) + (h2dot_y / PYi**NYi))

# def cbf3dot_lateral_inner(x,F):
#     h3dot_x = NXi*(NXi-1)*3*xdot(x)*x2dot(x,F) + \
#               NXi*(x[0] - CXi)**(NXi-1)*x3dot(x,F)
#     h3dot_y = NYi*(NYi-1)*3*ydot(x)*y2dot(x,F) + \
#               NYi*(x[1] - CYi)**(NYi-1)*y3dot(x,F)
#     return ((h3dot_x / PXi**NXi) + (h3dot_y / PYi**NYi))

# def cbf4dot_lateral_inner_uncontrolled(x,F):
#     h4dot_x = NXi*(NXi-1)*(3*x2dot(x,F)**2 + 4*xdot(x)*x3dot(x,F)) + \
#               NXi*(x[0] - CXi)**(NXi-1) * x4dot(x,F)[0]
#     h4dot_y = NYi*(NYi-1)*(3*y2dot(x,F)**2 + 4*ydot(x)*y3dot(x,F)) + \
#               NYi*(x[1] - CYi)**(NYi-1) * y4dot(x,F)[0]
#     return ((h4dot_x / PXi**NXi) + (h4dot_y / PYi**NYi))

# def cbf4dot_lateral_inner_controlled(x,F):
#     h4dot_x = NXi*(x[0] - CXi)**(NXi-1) * x4dot(x,F)[1]
#     h4dot_y = NYi*(x[1] - CYi)**(NYi-1) * y4dot(x,F)[1]
#     return ((h4dot_x / PXi**NXi) + (h4dot_y / PYi**NYi))

########################## Inner/Outer Lateral Safety ##########################

# def cbf_lateral_outer_0(x):
#     return 1 - ((x[0] - CXo)/PXo)**NXo - ((x[1] - CYo)/PYo)**NYo
#
# def cbfdot_lateral_outer_0(x):
#     hdot_x = NXo * (x[0] - CXo)**(NXo-1) / PXo**NXo * xdot(x)
#     hdot_y = NYo * (x[1] - CYo)**(NYo-1) / PYo**NYo * ydot(x)
#     return -((hdot_x / PXo**NXo) + (hdot_y / PYo**NYo))
#
# def cbf2dot_lateral_outer_0(x,F):
#     h2dot_x = (NXo * (NXo-1) * (x[0] - CXo)**(NXo-2) * xdot(x)**2) + \
#               (NXo*(x[0] - CXo)**(NXo-1) * x2dot(x,F))
#     h2dot_y = (NYo * (NYo-1) * (x[1] - CYo)**(NYo-2) * ydot(x)**2) + \
#               (NYo*(x[1] - CYo)**(NYo-1) * y2dot(x,F))
#     return -((h2dot_x / PXo**NXo) + (h2dot_y / PYo**NYo))
#
# def cbf_lateral_inner_0(x):
#     return ((x[0] - CXi)/PXi)**NXi + ((x[1] - CYi)/PYi)**NYi - 1
#
# def cbfdot_lateral_inner_0(x):
#     hdot_x = NXi * (x[0] - CXi)**(NXi-1) / PXi**NXi * xdot(x)
#     hdot_y = NYi * (x[1] - CYi)**(NYi-1) / PYi**NYi * ydot(x)
#     return ((hdot_x / PXi**NXi) + (hdot_y / PYi**NYi))
#
# def cbf2dot_lateral_inner_0(x,F):
#     h2dot_x = (NXi * (NXi-1) * xdot(x)**2) + \
#               (NXi*(x[0] - CXi)**(NXi-1) * x2dot(x,F))
#     h2dot_y = (NYi * (NYi-1) * ydot(x)**2) + \
#               (NYi*(x[1] - CYi)**(NYi-1) * y2dot(x,F))
#     return ((h2dot_x / PXi**NXi) + (h2dot_y / PYi**NYi))
#
# def cbf_lateral_outer(x):
#     # print("c1: {}".format(cbf_lateral_outer_0(x)))
#     # print("c2: {}".format(- SiLU(np.dot(rd_outer(x),Rot(x)[:,2]))))
#     q = np.dot(rd_outer(x),Rot(x)[:,2])
#     return cbf_lateral_outer_0(x) + SiLU(q)
#
# def cbfdot_lateral_outer(x):
#     # print("o1: {}".format(cbfdot_lateral_outer_0(x)))
#     # print("o2: {}".format(- np.dot(rddot_outer(x),Rot(x)[:,2])))
#     # print("o3: {}".format(- np.dot(rd_outer(x),RotDot(x)[:,2])))
#     # print("rddot: {}".format(rddot_outer(x)))
#     # print("R:     {}".format(Rot(x)[:,2]))
#     # print("rd:    {}".format(rd_outer(x)))
#     # print("Rdot:  {}".format(RotDot(x)[:,2]))
#     q    = np.dot(rd_outer(x),Rot(x)[:,2])
#     qdot = np.dot(rddot_outer(x),Rot(x)[:,2]) + np.dot(rd_outer(x),RotDot(x)[:,2])
#     # print("dot1: {}".format(cbfdot_lateral_outer_0(x)))
#     # print("dot2: {}".format(- SiLUdot(q,qdot)))
#     return cbfdot_lateral_outer_0(x) + SiLUdot(q,qdot)
#
# def cbf2dot_lateral_outer_uncontrolled(x,F):
#     q     = np.dot(rd_outer(x),Rot(x)[:,2])
#     qdot  = np.dot(rddot_outer(x),Rot(x)[:,2]) + np.dot(rd_outer(x),RotDot(x)[:,2])
#     q2dot = np.dot(rd2dot_outer(x,F),Rot(x)[:,2]) + np.dot(rd_outer(x),Rot2Dot_uncontrolled(x)[:,2]) + 2*np.dot(rddot_outer(x),RotDot(x)[:,2])
#     return cbf2dot_lateral_outer_0(x,F) + SiLU2dot_uncontrolled(q,qdot,q2dot)
#
# def cbf2dot_lateral_outer_controlled(x,F):
#     q     = np.dot(rd_outer(x),Rot(x)[:,2])
#     qdot  = np.dot(rddot_outer(x),Rot(x)[:,2]) + np.dot(rd_outer(x),RotDot(x)[:,2])
#     q2dot = np.dot(rd_outer(x),Rot2Dot_controlled(x)[:,2])
#     return +SiLU2dot_controlled(q,qdot,q2dot)
#
# def cbf_lateral_inner(x):
#     q = np.dot(rd_inner(x),Rot(x)[:,2])
#     return cbf_lateral_inner_0(x) + SiLU(q)
#
# def cbfdot_lateral_inner(x):
#     q    = np.dot(rd_inner(x),Rot(x)[:,2])
#     qdot = np.dot(rddot_inner(x),Rot(x)[:,2]) + np.dot(rd_inner(x),RotDot(x)[:,2])
#     return cbfdot_lateral_inner_0(x) + SiLUdot(q,qdot)
#
# def cbf2dot_lateral_inner_uncontrolled(x,F):
#     q     = np.dot(rd_inner(x),Rot(x)[:,2])
#     qdot  = np.dot(rddot_inner(x),Rot(x)[:,2]) + np.dot(rd_inner(x),RotDot(x)[:,2])
#     q2dot = np.dot(rd2dot_inner(x,F),Rot(x)[:,2]) + np.dot(rd_inner(x),Rot2Dot_uncontrolled(x)[:,2]) + 2*np.dot(rddot_inner(x),RotDot(x)[:,2])
#     return cbf2dot_lateral_inner_0(x,F) + SiLU2dot_uncontrolled(q,qdot,q2dot)
#
# def cbf2dot_lateral_inner_controlled(x,F):
#     q     = np.dot(rd_inner(x),Rot(x)[:,2])
#     qdot  = np.dot(rddot_inner(x),Rot(x)[:,2]) + np.dot(rd_inner(x),RotDot(x)[:,2])
#     q2dot = np.dot(rd_inner(x),Rot2Dot_controlled(x)[:,2])
#     return +SiLU2dot_controlled(q,qdot,q2dot)
#
# ############################# Attitude Safety #############################
#
# def cbf_attitude_0(x):
#     angle_in_deg = 90.0
#     return np.cos(x[6])*np.cos(x[7]) - np.cos(np.pi/180 * angle_in_deg)
#
# def cbfdot_attitude_0(x):
#     return -phidot(x)*np.sin(x[6])*np.cos(x[7]) - thedot(x)*np.cos(x[6])*np.sin(x[7])
#
# def cbf2dot_attitude_uncontrolled_0(x):
#     return -phi2dot(x)[0]*np.sin(x[6])*np.cos(x[7]) - the2dot(x)[0]*np.cos(x[6])*np.sin(x[7])\
#            -(phidot(x)**2 + thedot(x)**2)*np.cos(x[6])*np.cos(x[7]) + 2*phidot(x)*thedot(x)*np.sin(x[6])*np.sin(x[7])
#
# def cbf2dot_attitude_controlled_0(x):
#     return -phi2dot(x)[1]*np.sin(x[6])*np.cos(x[7]) - the2dot(x)[1]*np.cos(x[6])*np.sin(x[7])
#
# def cbf_attitude(x):
#     # return cbf_attitude_0(x)**3
#     return cbf_attitude_0(x)
#     return np.log(cbf_attitude_0(x) + 1)
#
# def cbfdot_attitude(x):
#     # return 3 * cbf_attitude_0(x)**2 * cbfdot_attitude_0(x)
#     return cbfdot_attitude_0(x)
#     return cbfdot_attitude_0(x) / (cbf_attitude_0(x) + 1)
#
# def cbf2dot_attitude_uncontrolled(x):
#     # return 6 * cbf_attitude_0(x) * cbfdot_attitude_0(x)**2 + 3 * cbf_attitude_0(x)**2 * cbf2dot_attitude_uncontrolled_0(x)
#     return cbf2dot_attitude_uncontrolled_0(x)
#     return (cbf2dot_attitude_uncontrolled_0(x) - cbfdot_attitude_0(x)**2) / (cbf_attitude_0(x) + 1)**2
#
# def cbf2dot_attitude_controlled(x):
#     # return 3 * cbf_attitude_0(x)**2 * cbf2dot_attitude_controlled_0(x)
#     return cbf2dot_attitude_controlled_0(x)
#     return cbf2dot_attitude_controlled_0(x) / (cbf_attitude_0(x) + 1)**2
#
# ############################# Velocity Safety #############################
#
# def cbf_velocity_outer(x):
#     D = PXo - dist(x)
#     return D - np.dot(vel(x),rd_outer(x))
#
# def cbfdot_velocity_outer(x,F):
#     print("dd: {}".format(-distdot(x)))
#     print("vdot: {}, rd_outer: {}".format(-veldot(x,F),rd_outer(x)))
#     print("2: {}".format(- np.dot(vel(x),rddot_outer(x))))
#     return -distdot(x) - np.dot(veldot(x,F),rd_outer(x)) - np.dot(vel(x),rddot_outer(x))
#
# def cbf2dot_velocity_outer(x,F):
#     return -dist2dot(x,F) - np.dot(vel2dot(x,F),rd_outer(x)) - np.dot(vel(x),rd2dot_outer(x,F)) - 2*np.dot(veldot(x,F),rddot_outer(x))
#
# def cbf3dot_velocity_outer_uncontrolled(x,F):
#     return -dist3dot(x,F) - np.dot(vel3dot_uncontrolled(x,F),rd_outer(x)) - np.dot(vel(x),rd3dot_outer(x,F)) - 3*np.dot(vel2dot(x,F),rddot_outer(x)) - 3*np.dot(veldot(x,F),rd2dot_outer(x,F))
#
# def cbf3dot_velocity_outer_controlled(x,F):
#     return - np.dot(vel3dot_controlled(x,F),rd_outer(x))
#
# def cbf_velocity_inner(x):
#     D = dist(x) - PXi
#     return D - np.dot(vel(x),rd_inner(x))
#
# def cbfdot_velocity_inner(x,F):
#     return distdot(x) - np.dot(veldot(x,F),rd_inner(x)) - np.dot(vel(x),rddot_inner(x))
#
# def cbf2dot_velocity_inner(x,F):
#     return dist2dot(x,F) - np.dot(vel2dot(x,F),rd_inner(x)) - np.dot(vel(x),rd2dot_inner(x,F)) - 2*np.dot(veldot(x,F),rddot_inner(x))
#
# def cbf3dot_velocity_inner_uncontrolled(x,F):
#     return dist3dot(x,F) - np.dot(vel3dot_uncontrolled(x,F),rd_inner(x)) - np.dot(vel(x),rd3dot_inner(x,F)) - 3*np.dot(vel2dot(x,F),rddot_inner(x)) - 3*np.dot(veldot(x,F),rd2dot_inner(x,F))
#
# def cbf3dot_velocity_inner_controlled(x,F):
#     return - np.dot(vel3dot_controlled(x,F),rd_outer(x))
#
# ############################# Velocity Helpers #############################
#
# def vel(x):
#     return np.array([xdot(x),ydot(x),zdot(x)])
#
# def veldot(x,F):
#     return np.array([x2dot(x,F),y2dot(x,F),np.sum(z2dot(x,F))])
#
# def vel2dot(x,F):
#     return np.array([x3dot(x,F),y3dot(x,F),z3dot(x,F)])
#
# def vel3dot_uncontrolled(x,F):
#     return np.array([x4dot(x,F)[0],y4dot(x,F)[0],z4dot(x,F)[0]])
#
# def vel3dot_controlled(x,F):
#     return np.array([x4dot(x,F)[1],y4dot(x,F)[1],z4dot(x,F)[1]])
#
# def rd_outer(x):
#     th     = theta(x)
#
#     return np.array([PXo*np.cos(th) - x[0],
#                      PYo*np.sin(th) -
#                       x[1],
#                      0.])
#
# def rddot_outer(x):
#     th     = theta(x)
#     thdot  = thetadot_outer(x)
#
#     return np.array([-PXo * np.sin(th) * thdot - xdot(x),
#                       PYo * np.cos(th) * thdot - ydot(x),
#                       0.])
#
# def rd2dot_outer(x,F):
#     th     = theta(x)
#     thdot  = thetadot_outer(x)
#     th2dot = theta2dot_outer(x,F)
#
#     return np.array([-PXo * np.cos(th) * thdot**2 - PXo * np.sin(th) * th2dot - x2dot(x,F),
#                      -PYo * np.sin(th) * thdot**2 + PYo * np.cos(th) * th2dot - y2dot(x,F),
#                      0.])
#
# def rd3dot_outer(x,F):
#     th     = theta(x)
#     thdot  = thetadot_outer(x)
#     th2dot = theta2dot_outer(x,F)
#     th3dot = theta3dot_outer(x,F)
#
#     return np.array([PXo * np.sin(th) * thdot**3 - 3*PXo * np.cos(th) * thdot * th2dot - PXo * np.sin(th) * th3dot - x3dot(x,F),
#                     -PYo * np.cos(th) * thdot**3 - 3*PYo * np.sin(th) * thdot * th2dot + PYo * np.cos(th) * th3dot - y3dot(x,F),
#                     0.])
#
# def rd_inner(x):
#     th     = theta(x)
#
#     return 1 * np.array([PXi*np.cos(th) - x[0],
#                           PYi*np.sin(th) - x[1],
#                           0.])
#
# def rddot_inner(x):
#     th     = theta(x)
#     thdot  = thetadot_inner(x)
#
#     return 1 * np.array([-PXi * np.sin(th) * thdot - xdot(x),
#                            PYi * np.cos(th) * thdot - ydot(x),
#                            0.])
#
# def rd2dot_inner(x,F):
#     th     = theta(x)
#     thdot  = thetadot_inner(x)
#     th2dot = theta2dot_inner(x,F)
#
#     return 1 * np.array([-PXi * np.cos(th) * thdot**2 - PXi * np.sin(th) * th2dot - x2dot(x,F),
#                           -PYi * np.sin(th) * thdot**2 + PYi * np.cos(th) * th2dot - y2dot(x,F),
#                            0.])
#
# def rd3dot_inner(x,F):
#     th     = theta(x)
#     thdot  = thetadot_inner(x)
#     th2dot = theta2dot_inner(x,F)
#     th3dot = theta3dot_inner(x,F)
#
#     return 1 * np.array([ PXi * np.sin(th) * thdot**3 - 3*PXi * np.cos(th) * thdot * th2dot - PXi * np.sin(th) * th3dot - x3dot(x,F),
#                           -PYi * np.cos(th) * thdot**3 - 3*PYi * np.sin(th) * thdot * th2dot + PYi * np.cos(th) * th3dot - y3dot(x,F),
#                            0.])
#
# def dist(x):
#     return np.sqrt(x[0]**2 + x[1]**2)
#
# def distdot(x):
#     return - (x[0]*xdot(x) + x[1]*ydot(x)) / np.sqrt(x[0]**2 + x[1]**2)
#
# def dist2dot(x,F):
#     return (x[0]*xdot(x) + x[1]*ydot(x))**2 * (x[0]**2 + x[1]**2)**(-3/2) - \
#            (x[0]*x2dot(x,F) + xdot(x)**2 + x[1]*y2dot(x,F) + ydot(x)**2) / np.sqrt(x[0]**2 + x[1]**2)
#
# def dist3dot(x,F):
#     return 2*(x[0]*xdot(x) + x[1]*ydot(x))*(x[0]*x2dot(x,F) + xdot(x)**2 + x[1]*y2dot(x,F) + ydot(x)**2)*(x[0]**2 + x[1]**2)**(-3/2) - \
#            3*(x[0]*xdot(x) + x[1]*ydot(x))**3*(x[0]**2 + x[1]**2)**(-5/2) + \
#            (x[0]*x2dot(x,F) + xdot(x)**2 + x[1]*y2dot(x,F) + ydot(x)**2) * (x[0]*xdot(x) + x[1]*ydot(x)) * (x[0]**2 + x[1]**2)**(-3/2) - \
#            (x[0]*x3dot(x,F) + 3*xdot(x)*x2dot(x,F) + x[1]*y3dot(x,F) + 3*ydot(x)*y2dot(x,F)) / np.sqrt(x[0]**2 + x[1]**2)
#
# def theta(x):
#     return np.arctan2(x[1]-CYo,x[0]-CXo)
#
# def thetadot_outer(x):
#     a = CXo
#     b = CYo
#
#     return ((x[0] - a)*ydot(x) + (b - x[1])*xdot(x))/(a**2 - 2*a*x[0] + b**2 - 2*b*x[1] + x[0]**2 + x[1]**2)
#
# def thetadot_inner(x):
#     a = CXi
#     b = CYi
#
#     return ((x[0] - a)*ydot(x) + (b - x[1])*xdot(x))/(a**2 - 2*a*x[0] + b**2 - 2*b*x[1] + x[0]**2 + x[1]**2)
#
# def theta2dot_outer(x,F):
#     a = CXo
#     b = CYo
#
#     return ((x[0] - a)*y2dot(x,F) + (b - x[1])*x2dot(x,F))/(a**2 - 2*a*x[0] + b**2 - 2*b*x[1] + x[0]**2 + x[1]**2) - (((x[0] - a)*ydot(x) + (b - x[1])*xdot(x))*(-2*a*xdot(x) - 2*b*ydot(x) + 2*x[0]*xdot(x) + 2*x[1]*ydot(x)))/(a**2 - 2*a*x[0] + b**2 - 2*b*x[1] + x[0]**2 + x[1]**2)**2
#
# def theta2dot_inner(x,F):
#     a = CXi
#     b = CYi
#
#     return ((x[0] - a)*y2dot(x,F) + (b - x[1])*x2dot(x,F))/(a**2 - 2*a*x[0] + b**2 - 2*b*x[1] + x[0]**2 + x[1]**2) - (((x[0] - a)*ydot(x) + (b - x[1])*xdot(x))*(-2*a*xdot(x) - 2*b*ydot(x) + 2*x[0]*xdot(x) + 2*x[1]*ydot(x)))/(a**2 - 2*a*x[0] + b**2 - 2*b*x[1] + x[0]**2 + x[1]**2)**2
#
# def theta3dot_outer(x,F):
#     a = CXo
#     b = CYo
#
#     return (2*((ydot(x))/(x[0] - a) - ((x[1] - b)*xdot(x))/(x[0] - a)**2)*((2*(x[1] - b)*ydot(x))/(x[0] - a)**2 - (2*(x[1] - b)**2*xdot(x))/(x[0] - a)**3)**2)/((x[1] - b)**2/(x[0] - a)**2 + 1)**3 - (2*((2*(x[1] - b)*ydot(x))/(x[0] - a)**2 - (2*(x[1] - b)**2*xdot(x))/(x[0] - a)**3)*(-((x[1] - b)*x2dot(x,F))/(x[0] - a)**2 + (2*(x[1] - b)*xdot(x)**2)/(x[0] - a)**3 - (2*xdot(x)*ydot(x))/(x[0] - a)**2 + (y2dot(x,F))/(x[0] - a)))/((x[1] - b)**2/(x[0] - a)**2 + 1)**2 - (((ydot(x))/(x[0] - a) - ((x[1] - b)*xdot(x))/(x[0] - a)**2)*(-(2*(x[1] - b)**2*x2dot(x,F))/(x[0] - a)**3 - (8*(x[1] - b)*xdot(x)*ydot(x))/(x[0] - a)**3 + (6*(x[1] - b)**2*xdot(x)**2)/(x[0] - a)**4 + (2*(x[1] - b)*y2dot(x,F))/(x[0] - a)**2 + (2*ydot(x)**2)/(x[0] - a)**2))/((x[1] - b)**2/(x[0] - a)**2 + 1)**2 + (-(x3dot(x,F)*(x[1] - b))/(x[0] - a)**2 - (6*(x[1] - b)*xdot(x)**3)/(x[0] - a)**4 + (6*(x[1] - b)*xdot(x)*x2dot(x,F))/(x[0] - a)**3 - (3*x2dot(x,F)*ydot(x))/(x[0] - a)**2 - (3*xdot(x)*y2dot(x,F))/(x[0] - a)**2 + (6*xdot(x)**2*ydot(x))/(x[0] - a)**3 + (y3dot(x,F))/(x[0] - a))/((x[1] - b)**2/(x[0] - a)**2 + 1)
#
# def theta3dot_inner(x,F):
#     a = CXi
#     b = CYi
#
#     return (2*((ydot(x))/(x[0] - a) - ((x[1] - b)*xdot(x))/(x[0] - a)**2)*((2*(x[1] - b)*ydot(x))/(x[0] - a)**2 - (2*(x[1] - b)**2*xdot(x))/(x[0] - a)**3)**2)/((x[1] - b)**2/(x[0] - a)**2 + 1)**3 - (2*((2*(x[1] - b)*ydot(x))/(x[0] - a)**2 - (2*(x[1] - b)**2*xdot(x))/(x[0] - a)**3)*(-((x[1] - b)*x2dot(x,F))/(x[0] - a)**2 + (2*(x[1] - b)*xdot(x)**2)/(x[0] - a)**3 - (2*xdot(x)*ydot(x))/(x[0] - a)**2 + (y2dot(x,F))/(x[0] - a)))/((x[1] - b)**2/(x[0] - a)**2 + 1)**2 - (((ydot(x))/(x[0] - a) - ((x[1] - b)*xdot(x))/(x[0] - a)**2)*(-(2*(x[1] - b)**2*x2dot(x,F))/(x[0] - a)**3 - (8*(x[1] - b)*xdot(x)*ydot(x))/(x[0] - a)**3 + (6*(x[1] - b)**2*xdot(x)**2)/(x[0] - a)**4 + (2*(x[1] - b)*y2dot(x,F))/(x[0] - a)**2 + (2*ydot(x)**2)/(x[0] - a)**2))/((x[1] - b)**2/(x[0] - a)**2 + 1)**2 + (-(x3dot(x,F)*(x[1] - b))/(x[0] - a)**2 - (6*(x[1] - b)*xdot(x)**3)/(x[0] - a)**4 + (6*(x[1] - b)*xdot(x)*x2dot(x,F))/(x[0] - a)**3 - (3*x2dot(x,F)*ydot(x))/(x[0] - a)**2 - (3*xdot(x)*y2dot(x,F))/(x[0] - a)**2 + (6*xdot(x)**2*ydot(x))/(x[0] - a)**3 + (y3dot(x,F))/(x[0] - a))/((x[1] - b)**2/(x[0] - a)**2 + 1)
#
# ############################# SiLU Helpers #############################
# KSILU = 5.0
# def SiLU(q):
#     return KSILU * (q / (1 + np.exp(-q)))
#
# def SiLUdot(q,qdot):
#     return KSILU * (qdot * ( (1 + np.exp(-q) + q*np.exp(-q)) / (1 + np.exp(-q))**2 ))
#
# def SiLU2dot_uncontrolled(q,qdot,q2dot):
#     return KSILU * (q2dot*((1 + np.exp(-q) + q*np.exp(-q))/(1 + np.exp(-q))**2) + qdot**2*((np.exp(-q) + np.exp(-2*q) + 2*q*np.exp(-2*q))/(1 + np.exp(-q))**3))
#
# def SiLU2dot_controlled(q,qdot,q2dot):
#     return KSILU * (q2dot*((1 + np.exp(-q) + q*np.exp(-q))/(1 + np.exp(-q))**2))



# def cbf_velocity(x):
#     # return cbf_attitude_0(x)**3
#     return cbf_attitude_0(x)
#     return np.log(cbf_attitude_0(x) + 1)

# def cbfdot_velocity(x):
#     # return 3 * cbf_attitude_0(x)**2 * cbfdot_attitude_0(x)
#     return cbfdot_attitude_0(x)
#     return cbfdot_attitude_0(x) / (cbf_attitude_0(x) + 1)

# def cbf2dot_velocity_uncontrolled(x):
#     # return 6 * cbf_attitude_0(x) * cbfdot_attitude_0(x)**2 + 3 * cbf_attitude_0(x)**2 * cbf2dot_attitude_uncontrolled_0(x)
#     return cbf2dot_attitude_uncontrolled_0(x)
#     return (cbf2dot_attitude_uncontrolled_0(x) - cbfdot_attitude_0(x)**2) / (cbf_attitude_0(x) + 1)**2

# def cbf2dot_velocity_controlled(x):
#     # return 3 * cbf_attitude_0(x)**2 * cbf2dot_attitude_controlled_0(x)
#     return cbf2dot_attitude_controlled_0(x)
#     return cbf2dot_attitude_controlled_0(x) / (cbf_attitude_0(x) + 1)**2

############################# Details #############################

# nCBFs = cbf(z0).shape[0]