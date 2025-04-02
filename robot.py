from enum import Enum
import numpy as np
import math
from config import Config

class Action(Enum):
    INCREASE_RIGHT = 0,
    DECREASE_RIGHT = 1,
    INCREASE_LEFT = 2,
    DECREASE_LEFT = 3,
    BREAK = 4,
    NOTHING = 5

class Robot:
    def __init__(self, x, y, theta):
        # True robot state (ground truth)
        self.x = x
        self.y = y
        self.theta = theta  # in radians
        self.radius = 10.0  # for drawing the robot

        # Wheel configuration
        self.max_speed = 20
        self.dv = 1  # pixels per second
        self.wheel_distance = 20.  # distance between wheels in pixels
        self. right_velocity = 0
        self.left_velocity = 0


        # Sensor configuration: simulating 3 sensors (front, left, right)
        self.sensor_range = 100.0  # max range in pixels
        self.sensor_angles = [(2. * math.pi / 12) * i for i in range(12)]  # relative sensor directions

    def update(self, dt, maze):
        """
        Update the robot's true pose using axis-separated collision detection.
        This allows the robot to slide along walls.
        """
        # Calculate linear and angular velocity from wheel velocities
        linear_velocity = self.get_linear_velocity()
        angular_velocity = self.get_angular_velocity()

        # Update x-axis component
        proposed_x = self.x + linear_velocity * math.cos(self.theta) * dt
        cell_x = int(proposed_x // Config.CELL_SIZE)
        cell_y = int(self.y // Config.CELL_SIZE)
        if Config.GRID_WIDTH > cell_x >= 0 == maze[cell_y, cell_x] and 0 <= cell_y < Config.GRID_HEIGHT:
            self.x = proposed_x

        # Update y-axis component
        proposed_y = self.y + linear_velocity * math.sin(self.theta) * dt
        cell_x = int(self.x // Config.CELL_SIZE)
        cell_y = int(proposed_y // Config.CELL_SIZE)
        if Config.GRID_WIDTH > cell_x >= 0 == maze[cell_y, cell_x] and 0 <= cell_y < Config.GRID_HEIGHT:
            self.y = proposed_y

        # Update orientation
        self.theta += angular_velocity * dt
        self.theta %= 2 * math.pi

    def set_velocity(self, action: Action):
        """
        Set robot's right and left wheel velocities.
        """
        if action == Action.INCREASE_RIGHT:
            self.right_velocity = min(self.right_velocity + self.dv, self.max_speed)
        elif action == Action.DECREASE_RIGHT:
            self.right_velocity = max(self.right_velocity - self.dv, -self.max_speed)
        elif action == Action.INCREASE_LEFT:
            self.left_velocity = min(self.left_velocity + self.dv, self.max_speed)
        elif action == Action.DECREASE_LEFT:
            self.left_velocity = max(self.left_velocity - self.dv, -self.max_speed)
        elif action == Action.BREAK:
            self.left_velocity = 0
            self.right_velocity = 0
        elif action == Action.NOTHING:
            pass

    def get_linear_velocity(self):
        return (self.right_velocity + self.left_velocity) / 2

    def get_angular_velocity(self):
        return (self.right_velocity - self.left_velocity) / self.wheel_distance

    def get_pose(self):
        return np.array([self.x, self.y, self.theta])

    def get_sensor_readings(self, maze):
        """
            Simulate sensor readings using ray-casting.
            Gaussian noise is added to each measurement.
            Returns a list of distances (one per sensor).
        """
        readings = []
        noise_std = 1.0
        for angle in self.sensor_angles:
            sensor_angle = self.theta + angle
            distance = 0
            while distance < self.sensor_range:
                test_x = int((self.x + distance * math.cos(sensor_angle)) // Config.CELL_SIZE)
                test_y = int((self.y + distance * math.sin(sensor_angle)) // Config.CELL_SIZE)
                if test_x < 0 or test_x >= Config.GRID_WIDTH or test_y < 0 or test_y >= Config.GRID_HEIGHT:
                    break
                if maze[test_y, test_x] == 1:
                    break
                distance += 1
            noisy_distance = max(0, distance + np.random.normal(0, noise_std))
            readings.append(noisy_distance)
        return readings