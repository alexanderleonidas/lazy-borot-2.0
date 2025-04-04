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
        self.radius = 10.  # for drawing the robot

        # Wheel configuration
        self.max_speed = 20
        self.dv = 1  # pixels per second
        self.wheel_distance = self.radius*2  # distance between wheels in pixels
        self. right_velocity = 0
        self.left_velocity = 0

        # Sensor configuration: simulating 3 sensors (front, left, right)
        self.sensor_range = 80.0  # max range in pixels
        self.sensor_angles = [(2. * math.pi / 12) * i for i in range(12)]  # relative sensor directions


    def update_motion(self, dt, maze):
        if self.right_velocity == self.left_velocity:
            dx = self.right_velocity * math.cos(self.theta) * dt
            dy = self.right_velocity * math.sin(self.theta) * dt
            self.x, self.y = self.x + dx, self.y + dy
        elif self.right_velocity == -self.left_velocity:
            omega = self.get_angular_velocity()
            self.theta = self.theta + omega * dt
        else:
            signed_dist = self.radius * ((self.left_velocity + self.right_velocity) / (self.right_velocity - self.left_velocity))
            omega = self.get_angular_velocity()
            icc_pos = [self.x - signed_dist * math.sin(self.theta), self.y + signed_dist * math.cos(self.theta)]
            rotation_matrix = np.array([[math.cos(omega * dt), -math.sin(omega * dt), 0],
                                        [math.sin(omega * dt), math.cos(omega * dt), 0],
                                        [0, 0, 1]])
            translation_matrix = np.array([self.x - icc_pos[0], self.y - icc_pos[1], self.theta])
            add_matrix = np.array([icc_pos[0], icc_pos[1], omega*dt])

            x_prime, y_prime, theta_prime = (rotation_matrix.dot(translation_matrix) + add_matrix)

            self.x = self.x + x_prime
            self.y = self.y + y_prime
            self.theta = self.theta + theta_prime

            self._check_collision(self.x, self.y, dt, maze)

    def _check_collision(self, proposed_x, proposed_y, dt, maze):
        """
        Improved collision detection and handling using a circular bounding object
        and proper velocity decomposition along walls.
        """
        # Get current grid cell and check if it's valid
        current_cell_x = int(self.x // Config.CELL_SIZE)
        current_cell_y = int(self.y // Config.CELL_SIZE)

        # Check all nearby cells that could potentially intersect with the robot
        radius_in_cells = math.ceil(self.radius / Config.CELL_SIZE) + 1
        collision_detected = False
        collision_normal = np.array([0.0, 0.0])

        # Check all cells that might intersect with the robot's circular body
        for dy in range(-radius_in_cells, radius_in_cells + 1):
            for dx in range(-radius_in_cells, radius_in_cells + 1):
                cell_x = current_cell_x + dx
                cell_y = current_cell_y + dy

                # Skip invalid cells or non-wall cells
                if (cell_x < 0 or cell_x >= Config.GRID_WIDTH or
                        cell_y < 0 or cell_y >= Config.GRID_HEIGHT or
                        maze[cell_y, cell_x] == 0):
                    continue

                # Calculate distance between robot center and this cell's edges
                # Convert cell to world coordinates (corners)
                cell_left = cell_x * Config.CELL_SIZE
                cell_right = (cell_x + 1) * Config.CELL_SIZE
                cell_top = cell_y * Config.CELL_SIZE
                cell_bottom = (cell_y + 1) * Config.CELL_SIZE

                # Find closest point on the cell to the robot center
                closest_x = max(cell_left, min(proposed_x, cell_right))
                closest_y = max(cell_top, min(proposed_y, cell_bottom))

                # Calculate distance to closest point
                dx = proposed_x - closest_x
                dy = proposed_y - closest_y
                distance = math.sqrt(dx * dx + dy * dy)

                # Check for collision (distance < radius)
                if distance < self.radius:
                    collision_detected = True
                    # Calculate normal vector (from cell to robot)
                    if distance > 0:
                        normal = np.array([dx, dy]) / distance
                    else:  # Rare case of direct center overlap
                        # Use movement direction as a fallback
                        movement = np.array([proposed_x - self.x, proposed_y - self.y])
                        movement_mag = np.linalg.norm(movement)
                        normal = movement / movement_mag if movement_mag > 0 else np.array([1.0, 0.0])

                    collision_normal += normal

        if collision_detected:
            # Normalize the collision normal if we have multiple collision points
            norm = np.linalg.norm(collision_normal)
            if norm > 0:
                collision_normal = collision_normal / norm

            # Calculate velocity vector
            velocity = np.array([
                self.get_linear_velocity() * math.cos(self.theta),
                self.get_linear_velocity() * math.sin(self.theta)
            ])

            # Decompose velocity into parallel and perpendicular components
            v_perp = np.dot(velocity, collision_normal) * collision_normal
            v_parallel = velocity - v_perp

            # Only apply the parallel component of the velocity
            if np.linalg.norm(v_parallel) > 0:
                move_direction = v_parallel / np.linalg.norm(v_parallel)
                move_magnitude = np.linalg.norm(v_parallel) * dt

                # Update position using only the parallel component
                self.x += move_direction[0] * move_magnitude
                self.y += move_direction[1] * move_magnitude

            # Optionally adjust the robot's velocities based on collision
            # This would simulate a "bounce" or friction effect
            # For simplicity, we're just canceling the perpendicular component

            # Update orientation based on the adjusted movement
            self.theta += self.get_angular_velocity() * dt
            self.theta %= 2 * math.pi
        else:
            # No collision, apply full movement
            self.x = proposed_x
            self.y = proposed_y
            self.theta += self.get_angular_velocity() * dt
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