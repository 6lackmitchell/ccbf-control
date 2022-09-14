""" experiment.py: entry-point for Aion rover experiment. """

import numpy as np
from typing import List
import nptyping as npt
import builtins
from dasc_robots.robot import Robot
from dasc_robots.ros_functions import *
from core.agent import Agent
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from transformations import euler_from_quaternion


def get_robot_states(robots: List[Robot]) -> npt.NDArray:
    """Given a list of Robot objects, get all states and return as one numpy array.

    ARGUMENTS
    ---------
    robots: list of Robot objects

    RETURNS
    -------
    states: Nxn array of N robots containing n states

    """
    states = np.zeros((len(robots), 5))
    for rr, robot in enumerate(robots):
        pos = robot.get_world_position()
        vel = robot.get_world_velocity()
        phi = euler_from_quaternion(robot.get_body_quaternion())[2]

        states[rr, :] = np.array([pos[0], pos[1], phi, np.linalg.norm(vel), 0.0])

    return states


class MinimalPublisher(Node):

    def __init__(self,
                 robots: List[Robot],
                 agents: List[Agent]):
        super().__init__('minimal_publisher')
        self.robots = robots
        self.agents = agents
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Settings
        nTimesteps = 1000
        nStates = 5
        nControls = 2

        # Logging variables
        self.i = 0
        self.z = np.zeros((nTimesteps, len(robots), nStates))
        self.u = np.zeros((nTimesteps, len(robots), nControls))

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

        z = get_robot_states(self.robots)
        self.z_log[self.i] = z

        # Compute control inputs for all agents in the system
        for aa, (agent, robot) in enumerate(zip(self.agents, self.robots)):
            code, status = agent.compute_control(z)

            if not code:
                print('Error in Agent {}'.format(aa + 1))
                break

            # Compute velocity command control inputs
            omg = agent.controller.u[0]
            vel = z[aa, 3] + self.timer_period * agent.controller.u[1]

            print("Omega: {:.2f}! (Commanding 0)".format(omg))
            print("Veloc: {:.2f}! (Commanding 0)".format(vel))

            omg = 0.0
            vel = 0.0

            self.robots[aa].command_velocity(np.array([0, vel, 0, 0, omg]))


def main(args=None):
    rclpy.init(args=args)

    ros_init("test_run")

    robot3 = Robot("rover3", 3)
    robot4 = Robot("rover4", 4)
    print("Robot Initialized")

    robot3.init()
    robot4.init()

    robots = [robot3, robot4]

    robot3.set_command_mode('velocity')
    robot3.cmd_offboard_mode()
    robot3.arm()

    threads = start_ros_nodes(robots)

    minimal_publisher = MinimalPublisher(robots)

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

    print("hello")


def experiment(tf: float,
               dt: float,
               vehicle: str,
               level: str,
               situation: str) -> bool:
    """Simulates the system specified by config.py for tf seconds at a frequency of dt.

    ARGUMENTS
        tf: final time (in sec)
        dt: timestep length (in sec)
        vehicle: the vehicle to be simulated
        level: the control level (i.e. kinematic, dynamic, etc.)
        situation: i.e. intersection_old, intersection, etc.

    RETURNS
        None

    """
    # Program-wide specifications
    builtins.PROBLEM_CONFIG = {'vehicle': vehicle,
                               'control_level': level,
                               'situation': situation,
                               'system_model': 'deterministic'}

    if vehicle == 'bicycle':
        from bicycle import nAgents, nStates, z0, centralized_agents, decentralized_agents

    broken = False
    nTimesteps = int((tf - 0.0) / dt) + 1

    # Simulation setup
    z = np.zeros((nTimesteps, nAgents, nStates))
    complete = np.zeros((nAgents,))

    # ROS Setup
    rclpy.init(args=None)
    ros_init("C-CBF Rover Experiment")

    # Initialize Robots
    robots = [Robot("rover{}".format(rr), rr) for rr in [3, 5, 7]]
    for rr, robot in enumerate(robots):
        robot.init()
        robot.set_command_mode('velocity')
        robot.cmd_offboard_mode()
        robot.arm()
        print("Robot{} Armed".format(rr))

    # Start ROS Nodes
    threads = start_ros_nodes(robots)

    # Set up publishers
    minimal_publisher = MinimalPublisher(robots, agents)
    rclpy.spin(minimal_publisher)

    # Destroy
    minimal_publisher.destroy_node()
    rclpy.shutdown()


    # Simulate program
    for ii, tt in enumerate(np.linspace(0, tf, nTimesteps - 1)):
        code = 0
        if round(tt, 4) % 1 < dt:
            print("Time: {:.1f} sec".format(tt))

        if round(tt, 4) % 5 < dt and tt > 0:
            for aa, agent in enumerate(decentralized_agents):
                agent.save_data(aa)
            print("Time: {:.1f} sec: Intermediate Save".format(tt))

        # Compute inputs for centralized agents
        if centralized_agents is not None:
            centralized_agents.compute_control(tt, z[ii])

        # Iterate over all agents in the system
        for aa, agent in enumerate(decentralized_agents):
            code, status = agent.compute_control(z[ii])

            if not code:
                broken = True
                print('Error in Agent {}'.format(aa + 1))
                break
            if hasattr(agent, 'complete'):
                if agent.complete and not complete[aa]:
                    complete[aa] = True
                    print("Agent {} Completed!".format(aa))

            # Step dynamics forward
            z[ii + 1, aa, :] = agent.step_dynamics()

        if not code:
            broken = True
            break

        if np.sum(complete) == nAgents:
            break

    # Save data
    for aa, agent in enumerate(decentralized_agents):
        agent.save_data(aa)

    success = not broken

    return success


if __name__ == "__main__":
    end_time = 40.0
    timestep = 0.05
    veh = 'bicycle'
    lev = 'dynamic'
    sit = 'dasclab'

    success = experiment(end_time, timestep, veh, lev, sit)