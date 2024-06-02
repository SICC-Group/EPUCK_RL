# coding = utf-8
from deepbots.robots.controllers.csv_robot import CSVRobot
import os
import numpy as np
import sys
sys.path.append('..')
from supervisor_controller.config import get_config
args = get_config().parse_known_args()[0]
class Epuck2Robot(CSVRobot):
    def __init__(self):
        super().__init__(timestep=args.timestep)
        '''self.left_wheel_sensor = self.getDevice("left wheel sensor")
        self.right_wheel_sensor = self.getDevice("right wheel sensor")
        self.left_wheel_sensor.enable(self.timestep)
        self.right_wheel_sensor.enable(self.timestep)'''
        self.num_agents = args.num_agents
        self.ps_sensor = []
        for i in range(8):
            self.ps_sensor.append(self.getDevice(f"ps{i}"))
            self.ps_sensor[i].enable(self.timestep)

        self.wheels = []
        for wheel_name in ['left wheel motor', 'right wheel motor']:
            wheel = self.getDevice(wheel_name)  # Get the wheel handle
            wheel.setPosition(float('inf'))  # Set starting position
            wheel.setVelocity(0.0)  # Zero out starting velocity
            self.wheels.append(wheel)

        self.weights = [[-1.3, -1.0], [-1.3, -1.0], [-0.5, 0.5], [0.0, 0.0],
         [0.0, 0.0], [0.05, -0.5], [-0.75, 0], [-0.75, 0]]
        self.max_speed = 6.28
        self.robot_name = self.getName()[-1]

    def create_message(self):
        # Read the sensor value, convert to string and save it in a list
        message = []
        message.append('t'+self.robot_name)
        for rangefinder in self.ps_sensor:
            message.append(rangefinder.getValue())
        return message

    def handle_emitter(self):
        return None

    def use_message_data(self, message):
        action = int(message[int(self.robot_name)+self.num_agents-1])   # Convert the string message into an action integer
        speed = np.zeros(2)
        # if action == 1:
        #     rand = random.random()
        #     if rand <= 0.333:
        #         speed[0] = 0.75 * self.max_speed
        #         speed[1] = 0.25 * self.max_speed
        #     elif rand <= 0.666:
        #         speed[0] = 0.25 * self.max_speed
        #         speed[1] = 0.75 * self.max_speed
        #     else:
        #         speed[0] = self.max_speed / 3
        #         speed[1] = self.max_speed / 3
        # right_obstacle = ((self.ps_sensor[0].getValue() > 80.0) | (self.ps_sensor[1].getValue() > 80.0) | (self.ps_sensor[2].getValue() > 80.0))
        # left_obstacle = ((self.ps_sensor[5].getValue() > 80.0) | (self.ps_sensor[6].getValue() > 80.0) | (self.ps_sensor[7].getValue() > 80.0))
        # if left_obstacle:
        #     speed[0] = 0.5 * self.max_speed
        #     speed[1] = -0.5 * self.max_speed
        # elif right_obstacle:
        #     speed[0] = -0.5 * self.max_speed
        #     speed[1] = 0.5 * self.max_speed
        vel_actions = [[0, -1.57], [0, -0.785], [0, 0], [0, 0.785], [0, 1.57], [0.1, -1.57], [0.1, -0.785], [0.1, 0],
                       [0.1, 0.785], [0.1, 1.57], [0, 0]]
        linear_vel = vel_actions[action][0]
        angle_vel = vel_actions[action][1]
        speed[0] = ((2 * linear_vel) - (angle_vel * 0.053)) / (2 * 0.0205)
        speed[1] = ((2 * linear_vel) + (angle_vel * 0.053)) / (2 * 0.0205)

        speed = np.clip(speed,-self.max_speed,self.max_speed)
        if action == 0:
            speed[:] = 0
        # Set the motors' velocities based on the action received
        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(speed[i])


# Create the robot controller object and run it
robot_controller = Epuck2Robot()
robot_controller.run()  # Run method is implemented by the framework, just need to call it
