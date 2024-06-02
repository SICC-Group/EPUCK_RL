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
        #args = get_config().parse_known_args()[0]
        self.interval = args.interval
        self.weights = [[-1.3, -1.0], [-1.3, -1.0], [-0.5, 0.5], [0.0, 0.0],
                        [0.0, 0.0], [0.05, -0.5], [-0.75, 0], [-0.75, 0]]
        self.max_speed = 6.28
        self.robot_name = self.getName()[-1]
        self.timestep = args.timestep
        self.num_agents = args.num_agents

        self.ps_sensor = []
        for i in range(8):
            self.ps_sensor.append(self.getDevice(f"ps{i}"))
            self.ps_sensor[i].enable(self.timestep)

        self.accelerometer = self.getDevice("accelerometer")
        self.accelerometer.enable(self.timestep)

        self.wheels = []
        for wheel_name in ['left wheel motor', 'right wheel motor']:
            wheel = self.getDevice(wheel_name)  # Get the wheel handle
            wheel.setPosition(float('inf'))  # Set starting position
            wheel.setVelocity(0.0)  # Zero out starting velocity
            self.wheels.append(wheel)

        self.receiver_arb = self.initialize_receiver()


    def initialize_receiver(self):
        """
        This method implements the basic emitter/receiver initialization that
        assumes that an emitter and a receiver component are present on the
        Webots robot with appropriate DEFs ("emitter"/"receiver").

        :param emitter_name: The name of the emitter device on the
            supervisor node
        :param receiver_name: The name of the receiver device on the
            supervisor node
        :return: The initialized emitter and receiver references
        """
        receiver = self.getDevice('receiver_rab')
        receiver.enable(self.timestep//self.interval)
        return receiver

    def handle_receiver_arb(self):
        # print("supervisor",self.receiver.getQueueLength())
        message = 0
        if self.receiver_arb.getQueueLength() > 0:
            str_message = self.receiver_arb.getString()
            #message = np.array(self.receiver_arb.getFloats()).reshape(self.num_agents,-1)
            message = np.array(str_message.split(","),dtype=np.float32).reshape(self.num_agents,-1)
            #print(message)
            self.receiver_arb.nextPacket()
        return message

    def run(self):
        """
        This method is required by Webots to update the robot in the
        simulation. It steps the robot and in each step it runs the two
        handler methods to use the emitter and receiver components.

        This method should be called by a robot manager to run the robot.
        """
        i = 0
        while self.step(self.timestep//self.interval) != -1:
            self.handle_receiver_arb()
            if i%self.interval == 0:
                self.handle_receiver()
                self.handle_emitter()
                i = 0
            i += 1
            #print(i)
        #print(-1)

    def create_message(self):
        # Read the sensor value, convert to string and save it in a list
        message = []
        message.append('a'+self.robot_name)
        #message.append(self.accelerometer.getValues())
        for rangefinder in self.ps_sensor:
            message.append(rangefinder.getValue())
        return message

    def use_message_data(self, message):
        #print(self.robot_name)
        action = int(message[int(self.robot_name)-1])  # Convert the string message into an action integer
        speed = np.zeros(2)
        vel_actions = [[0,-1.57],[0,-0.785],[0,0],[0,0.785],[0,1.57],[0.1,-1.57],[0.1,-0.785],[0.1,0],[0.1,0.785],[0.1,1.57],[0,0]]
        linear_vel = vel_actions[action][0]
        angle_vel = vel_actions[action][1]
        speed[0] = ((2 * linear_vel) - (angle_vel * 0.053)) / (2 * 0.0205)
        speed[1] = ((2 * linear_vel) + (angle_vel * 0.053)) / (2 * 0.0205)
        '''        if action == 0:
            speed[0] = 1 * self.max_speed
            speed[1] = 0.2 * self.max_speed
        elif action == 1:
            speed[0] = 0.2 * self.max_speed
            speed[1] = 1 * self.max_speed
        elif action == 2:
            speed[0] = self.max_speed
            speed[1] = self.max_speed
        elif action == 3:
            speed[0] = 0.5 * self.max_speed
            speed[1] = -0.5 * self.max_speed
        elif action == 4:
            speed[0] = -0.5 * self.max_speed
            speed[1] = 0.5 * self.max_speed
        else:
            speed[0] = 0
            speed[1] = 0'''
        '''right_obstacle = ((self.ps_sensor[0].getValue() > 80.0) | (self.ps_sensor[1].getValue() > 80.0) | (self.ps_sensor[2].getValue() > 80.0))
        left_obstacle = ((self.ps_sensor[5].getValue() > 80.0) | (self.ps_sensor[6].getValue() > 80.0) | (self.ps_sensor[7].getValue() > 80.0))
        if left_obstacle:
            speed[0] = 0.5 * self.max_speed
            speed[1] = -0.5 * self.max_speed
        elif right_obstacle:
            speed[0] = -0.5 * self.max_speed
            speed[1] = 0.5 * self.max_speed'''

        speed = np.clip(speed, -self.max_speed, self.max_speed)
        '''if action == 4:
            speed[:] = 0'''
        # Set the motors' velocities based on the action received
        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(speed[i])


# Create the robot controller object and run it
robot_controller = Epuck2Robot()
robot_controller.run()  # Run method is implemented by the framework, just need to call it
