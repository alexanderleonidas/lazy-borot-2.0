import numpy as np
import math
from config import Config

class OccupancyGrid:
    """
    Represents an occupancy grid for mapping and localization in robotics.

    Provides a grid-based representation of the environment using
    log-odds to model the likelihood of cells being occupied or free,
    based on sensor data and robot pose. Enables probabilistic updating and
    conversion to different representations like probability or grayscale.
    """
    def __init__(self, robot):
        self.robot = robot
        self.width = Config.GRID_WIDTH
        self.height = Config.GRID_HEIGHT
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)

        p_occ = 0.85
        self.log_odds_occupied = np.log(p_occ / (1 - p_occ))
        self.log_odds_free = np.log((1 - p_occ) / p_occ)

        self.cell_size_pixels = Config.CELL_SIZE
        self.angular_tolerance_rad = np.radians(10.0)
        self.max_sensor_range_pixels = robot.sensor_range

        self.log_odds_min_val = -10.0
        self.log_odds_max_val = 10.0

        # Pre-calculate constants
        self.cell_size_quarter = self.cell_size_pixels / 4.0
        self.cell_size_half = self.cell_size_pixels / 2.0

    def update(self, robot_pose_pixels, sensor_readings):
        """
        Updates the occupancy grid map by incorporating sensor readings into the grid. The method
        uses the robot's current pose and sensor readings to update probabilities for both free
        and occupied cells via a ray-tracing algorithm. Cells along the path traced by the sensor
        beams are updated as free, and the endpoint of the beam is updated as occupied, provided
        the measurement is within sensor range.
        """
        robot_x, robot_y, robot_theta = robot_pose_pixels

        for measured_dist, relative_beam_angle in sensor_readings:
            global_beam_angle = robot_theta + relative_beam_angle
            cos_angle = math.cos(global_beam_angle)
            sin_angle = math.sin(global_beam_angle)

            is_max_range_reading = measured_dist >= self.max_sensor_range_pixels - self.cell_size_quarter
            trace_length = self.max_sensor_range_pixels if is_max_range_reading else measured_dist

            # Ray-trace for free cells
            current_dist = self.cell_size_half
            while current_dist < trace_length - self.cell_size_quarter:
                px = robot_x + current_dist * cos_angle
                py = robot_y + current_dist * sin_angle

                grid_col_idx = int(px / self.cell_size_pixels)
                grid_row_idx = int(py / self.cell_size_pixels)

                if 0 <= grid_row_idx < self.height and 0 <= grid_col_idx < self.width:
                    self.grid[grid_row_idx, grid_col_idx] += self.log_odds_free

                current_dist += self.cell_size_half
                if current_dist > self.max_sensor_range_pixels + self.cell_size_half:
                    break

            # Mark occupied cell
            if not is_max_range_reading and measured_dist < self.max_sensor_range_pixels:
                end_point_x = robot_x + measured_dist * cos_angle
                end_point_y = robot_y + measured_dist * sin_angle

                end_grid_col_idx = int(end_point_x / self.cell_size_pixels)
                end_grid_row_idx = int(end_point_y / self.cell_size_pixels)

                if 0 <= end_grid_row_idx < self.height and 0 <= end_grid_col_idx < self.width:
                    self.grid[end_grid_row_idx, end_grid_col_idx] += self.log_odds_occupied

        np.clip(self.grid, self.log_odds_min_val, self.log_odds_max_val, out=self.grid)

    def get_probability_grid(self):
        exp_log_odds = np.exp(self.grid)
        return exp_log_odds / (1 + exp_log_odds)

    def get_grayscale_grid(self):
        return ((1 - self.get_probability_grid()) * 255).astype(np.uint8)
