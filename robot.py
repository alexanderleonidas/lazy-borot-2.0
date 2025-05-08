from enum import Enum
import numpy as np
import math
from config import Config
from kalman_filter import KalmanFilter
from extended_kalman_filter import ExtendedKalmanFilter
from mapping import OccupancyGrid


class Action(Enum):
    INCREASE_RIGHT = 0,
    DECREASE_RIGHT = 1,
    INCREASE_LEFT = 2,
    DECREASE_LEFT = 3,
    BREAK = 4,
    NOTHING = 5


class Robot:
    def __init__(self, x, y, theta, filter_type=None, mapping=False, ann=False):
        # True robot state (ground truth)
        self.x = x
        self.y = y
        self.theta = theta  # in radians
        self.radius = 4.  # for drawing the robot

        # Extra Variables
        self.last_collision_cell = None  # stores (i, j) of last obstacle collision
        self.path_history = []
        self.num_collisions = 0
        # self.max_history_length = 200

        # Wheel configuration
        self.max_speed = 80
        self.dv = 4  # pixels per second
        self.wheel_distance = self.radius*2  # distance between wheels in pixels
        self. right_velocity = 0
        self.left_velocity = 0

        # Sensor configuration: simulating 3 sensors (front, left, right)
        self.sensor_range = 128.0  # max range in pixels
        self.sensor_angles = [(2. * math.pi / 12) * i for i in range(12)]  # relative sensor directions
        self.visible_measurements = []  # list of visible landmarks
        self.sensor_readings = []  # list of sensor readings
        self.get_sensor_readings(Config.maze_grid)

        # --- SLAM / Localization Components ---
        if filter_type == 'KF':
            self.filter = KalmanFilter(self)
        elif filter_type == 'EKF':
            self.filter = ExtendedKalmanFilter(self)

        if mapping:
            self.mapping = OccupancyGrid(self)

        if ann:
            self.ann = ann

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

        # Compute the new pose based on kinematics.
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

            self.num_collisions += 1

        # Finally, update the orientation.
        self.theta = new_theta
        # print(f'\rRobot Pose: x: {self.x:.2f} | y: {self.y:.2f} | θ: {self.theta:.2f}', end='', flush=True)


        # Add current position to path history
        self.path_history.append((self.x, self.y))
        # if len(self.path_history) > self.max_history_length:
        #     self.path_history.pop(0)

        self.get_sensor_readings(maze)


    def circle_collides(self, x, y, maze):
        """
        Check if a circle (with the center (x, y) and radius self.radius)
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

    def get_visible_landmark_readings(self):
        visible_measurements = []
        cell_size = Config.CELL_SIZE

        for lm_x, lm_y in Config.landmarks:
            dx = lm_x - self.x
            dy = lm_y - self.y
            distance = math.hypot(dx, dy)

            if distance <= self.sensor_range:
                # Check visibility using the Bresenham-like approach
                is_visible = True
                x, y = self.x, self.y
                step_dx = abs(dx)
                step_dy = abs(dy)
                err = step_dx - step_dy
                sx = 1 if dx > 0 else -1
                sy = 1 if dy > 0 else -1

                while not (abs(x - lm_x) < 1 and abs(y - lm_y) < 1):
                    cell_x = int(x // cell_size)
                    cell_y = int(y // cell_size)

                    if (0 <= cell_x < Config.GRID_WIDTH and
                            0 <= cell_y < Config.GRID_HEIGHT and
                            Config.maze_grid[cell_y, cell_x] == 1):
                        is_visible = False
                        break

                    e2 = 2 * err
                    if e2 > -step_dy:
                        err -= step_dy
                        x += sx
                    if e2 < step_dx:
                        err += step_dx
                        y += sy

                if is_visible:
                    bearing = math.atan2(dy, dx) - self.theta
                    z = np.array([distance, (bearing % (2 * math.pi))])
                    visible_measurements.append((z, np.array([lm_x, lm_y])))
        self.visible_measurements = visible_measurements

    def _cast_ray(self, start_x, start_y, angle_rad, max_range, maze):
        """
        Helper function to cast a single ray.
        Returns the distance to the first obstacle or max_range.
        """
        distance = 0
        # More robust step size, could be 1 or smaller for finer grids
        step_size = 1.0
        while distance < max_range:
            # Calculate point along the ray
            ray_x = start_x + distance * math.cos(angle_rad)
            ray_y = start_y + distance * math.sin(angle_rad)

            # Convert to grid cell
            test_x_cell = int(ray_x // Config.CELL_SIZE)
            test_y_cell = int(ray_y // Config.CELL_SIZE)

            # Check bounds
            if not (0 <= test_x_cell < Config.GRID_WIDTH and 0 <= test_y_cell < Config.GRID_HEIGHT):
                return max_range  # Hit boundary, treat as max range

            # Check for obstacle
            if maze[test_y_cell, test_x_cell] == 1:
                return distance  # Hit obstacle

            distance += step_size
        return max_range  # No obstacle found within max_range

    def get_sensor_readings(self, maze):
        """
        Simulate sensor readings using ray-casting (for ground truth simulation).
        This populates self.sensor_readings.
        """
        simulated_readings = []
        # Sensor noise for simulation (different from the likelihood model's sigma_hit)
        simulation_noise_std = 1.0  # Example: 1 pixel std dev for simulated readings

        for rel_angle_rad in self.sensor_angles:
            abs_sensor_angle = self.theta + rel_angle_rad
            true_distance = self._cast_ray(self.x, self.y, abs_sensor_angle, self.sensor_range, maze)

            # Add noise to the true distance to get a simulated measurement
            noisy_distance = true_distance + np.random.normal(0, simulation_noise_std)
            noisy_distance = max(0, min(noisy_distance, self.sensor_range))  # Clamp to [0, sensor_range]

            simulated_readings.append((noisy_distance, rel_angle_rad))
        self.sensor_readings = simulated_readings

    def set_velocity(self, action):
        """
        Set the robot's right and left-wheel velocities.

        :param action: Action to take
        :type action: Action
        """
        if self.ann and isinstance(action, int):
            # Assuming the ANN output (action) is an integer representing the index of the Action enum
            # We need to map this integer back to the corresponding Action enum member
            try:
                action = list(Action)[action]
            except ValueError:
                raise ValueError(f"Invalid action index received from ANN: {action}. Must be a valid Action enum index.")

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
        else:
            raise ValueError("Invalid action. Use Action enum or ANN for velocity control.")


    def get_ann_inputs(self):
        """
        Get the inputs for the ANN.
        This includes sensor readings, robot pose, and a bias term.
        """
        # TODO: think about using the localisation and mapping information as inputs for the ANN
        # Normalize sensor readings to [0, 1]
        normalized_readings = [(reading[0] / self.sensor_range) for reading in self.sensor_readings]
        # Normalize Robot pose
        pose = [self.x / Config.WINDOW_WIDTH, self.y / Config.WINDOW_HEIGHT, self.theta / (2 * math.pi)]
        # Combine with bias
        return normalized_readings + pose + [1.0]

    def get_speed(self):
        return (self.right_velocity + self.left_velocity) / 2

    def get_angular_velocity(self):
        return (self.right_velocity - self.left_velocity) / self.wheel_distance

    def get_pose(self):
        return np.array([self.x, self.y, self.theta])

    def get_distance_traveled(self):
        if len(self.path_history) < 2:
            return 0
        distance = 0
        for i in range(1, len(self.path_history)):
            x1, y1 = self.path_history[i - 1]
            x2, y2 = self.path_history[i]
            distance += math.hypot(x2 - x1, y2 - y1)
        return distance