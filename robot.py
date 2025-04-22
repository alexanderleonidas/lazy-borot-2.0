from enum import Enum
import numpy as np
import math
from config import Config
from kalman_filter import KalmanFilter


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
        self.radius = 10.  # for drawing the robot

        # Extra Variables
        self.last_collision_cell = None  # stores (i, j) of last obstacle collision
        self.path_history = []
        self.max_history_length = 200

        # Wheel configuration
        self.max_speed = 50
        self.dv = 5  # pixels per second
        self.wheel_distance = self.radius*2  # distance between wheels in pixels
        self. right_velocity = 0
        self.left_velocity = 0

        # Sensor configuration: simulating 3 sensors (front, left, right)
        self.sensor_range = 150.0  # max range in pixels
        self.sensor_angles = [(2. * math.pi / 12) * i for i in range(12)]  # relative sensor directions
        self.visible_measurements = []  # list of visible landmarks

        self.filter = KalmanFilter(self)

    def update_motion(self, dt, maze):
        """
        Update the robot’s pose using the exact differential drive motion model,
        with strict collision detection that checks the robot’s circular boundary.
        """
        # Compute linear and angular velocities.
        v_r = self.right_velocity
        v_l = self.left_velocity
        v = (v_r + v_l) / 2.0
        omega = (v_r - v_l) / self.wheel_distance

        # Compute new pose based on kinematics.
        if abs(omega) < 1e-6:
            # Straight-line motion (or very small rotation)
            new_x = self.x + v * math.cos(self.theta) * dt
            new_y = self.y + v * math.sin(self.theta) * dt
            new_theta = self.theta
        else:
            # Motion along an arc using the ICC method.
            r = v / omega  # Turning radius
            icc_x = self.x - r * math.sin(self.theta)
            icc_y = self.y + r * math.cos(self.theta)
            delta_theta = omega * dt
            cos_dt = math.cos(delta_theta)
            sin_dt = math.sin(delta_theta)
            new_x = cos_dt * (self.x - icc_x) - sin_dt * (self.y - icc_y) + icc_x
            new_y = sin_dt * (self.x - icc_x) + cos_dt * (self.y - icc_y) + icc_y
            new_theta = (self.theta + delta_theta) % (2 * math.pi)
        

        # --- Strict collision detection using the robot's circular footprint ---
        # First, try to update fully if the new position is free.
        if not self.circle_collides(new_x, new_y, maze):
            self.x, self.y = new_x, new_y
            self.last_collision_cell = None
        else:
            # Otherwise, perform axis-separated updates to allow sliding.
            if not self.circle_collides(new_x, self.y, maze):
                self.x = new_x
            if not self.circle_collides(self.x, new_y, maze):
                self.y = new_y

        # Finally, update the orientation.
        self.theta = new_theta
        #print(f'\rRobot Pose: x: {self.x:.2f} | y: {self.y:.2f} | θ: {self.theta:.2f}', end='', flush=True)

        # Add current position to path history
        self.path_history.append((self.x, self.y))
        if len(self.path_history) > self.max_history_length:
            self.path_history.pop(0)
        

    def circle_collides(self, x, y, maze):
        """
        Check if a circle (with center (x, y) and radius self.radius)
        collides with any obstacle in the maze or screen boundaries.
        Obstacle cells are assumed to have a value of 1.
        """
        # First, check collision with screen boundaries
        if (x - self.radius < 0 or
                x + self.radius > Config.WINDOW_WIDTH or
                y - self.radius < 0 or
                y + self.radius > Config.WINDOW_HEIGHT):
            return True

        # Then check collision with maze obstacles
        cell_size = Config.CELL_SIZE
        radius = self.radius
        # Determine the grid cells within the circle's bounding box.
        min_cell_x = int((x - radius) // cell_size)
        max_cell_x = int((x + radius) // cell_size)
        min_cell_y = int((y - radius) // cell_size)
        max_cell_y = int((y + radius) // cell_size)

        # Check each cell in the bounding box.
        for cell_y in range(min_cell_y, max_cell_y + 1):
            for cell_x in range(min_cell_x, max_cell_x + 1):
                if 0 <= cell_x < Config.GRID_WIDTH and 0 <= cell_y < Config.GRID_HEIGHT:
                    if maze[cell_y, cell_x] == 1:  # This cell is a wall.
                        # Get the cell's boundaries.
                        rect_left = cell_x * cell_size
                        rect_right = (cell_x + 1) * cell_size
                        rect_top = cell_y * cell_size
                        rect_bottom = (cell_y + 1) * cell_size

                        # Find the closest point on the cell's rectangle to the circle center.
                        closest_x = max(rect_left, min(x, rect_right))
                        closest_y = max(rect_top, min(y, rect_bottom))

                        # Calculate the distance from the circle's center to this closest point.
                        distance = math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
                        if distance < radius:
                            self.last_collision_cell = (cell_y, cell_x)
                            return True
        return False

    def get_sensor_readings(self, maze):
        """
            Simulate sensor readings using ray-casting.
            Gaussian noise is added to each measurement.
            Returns a list of distances (one per sensor).
        """
        readings = []
        noise_std = 0.1
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
    
    def get_right_wheel_velocity(self):
        return self.right_velocity
    
    def get_left_wheel_velocity(self):
        return self.left_velocity

    def get_angular_velocity(self):
        return (self.right_velocity - self.left_velocity) / self.wheel_distance

    def get_pose(self):
        return np.array([self.x, self.y, self.theta])
