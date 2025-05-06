import numpy as np
import math
from config import Config
from scipy.linalg import norm


class OccupancyGrid:
    def __init__(self, robot):
        self.robot = robot

        self.width = Config.GRID_WIDTH
        self.height = Config.GRID_HEIGHT
        # Initialize a grid with log-odds 0 (unknown probability 0.5)
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)

        # Log-odds values for updating cells
        # P(occupied | measurement) = 0.9 => log(0.9/0.1)
        # P(free | measurement) = 0.7 (so P(occupied | measurement) = 0.3) => log(0.3/0.7)
        self.log_odds_occupied = np.log(0.85 / (1 - 0.85))  # Tunable: e.g. 0.85 / 0.15
        self.log_odds_free = np.log((1 - 0.85) / 0.85)  # Tunable: e.g. 0.15 / 0.85

        # Renaming for clarity (pixel units)
        self.cell_size_pixels = Config.CELL_SIZE
        self.angular_tolerance_rad = 10.0 * np.pi / 180.0  # beta: 10 degrees, for vectorized version
        self.max_sensor_range_pixels = robot.sensor_range

        # Log-odds clamping limits
        self.log_odds_min_val = -10.0  # Corresponds to P_min approx 0.000045
        self.log_odds_max_val = 10.0  # Corresponds to P_max approx 0.99995

        # --- For Vectorized Approach  ---
        # xs = np.arange(self.width) * Config.CELL_SIZE  # pixel x-coordinates
        # ys = np.arange(self.height) * Config.CELL_SIZE  # pixel y-coordinates
        # # Cell centers
        # grid_x_centers, grid_y_centers = np.meshgrid(
        #     xs + Config.CELL_SIZE / 2.0,
        #     ys + Config.CELL_SIZE / 2.0,
        #     indexing="xy"
        # )
        # self.grid_cell_center_coords_pixels = np.stack((grid_x_centers, grid_y_centers))
        # --- End Vectorized Approach variables ---

    def update(self, robot_pose_pixels, sensor_readings_pixels):
        """
        Update the occupancy grid using an inverse sensor model.
        robot_pose_pixels: (x, y, theta) in pixel coordinates and radians.
        sensor_readings_pixels: list of (distance_pixels, relative_angle_rad)
        """
        robot_x, robot_y, robot_theta = robot_pose_pixels

        # --- Iterative Ray-Casting (Refined) ---
        for measured_dist, relative_beam_angle in sensor_readings_pixels:
            global_beam_angle = robot_theta + relative_beam_angle

            # Determine the length of the ray to trace for 'free' space.
            # If it's a max range reading, trace up to max_sensor_range_pixels.
            # Otherwise, trace up to the measured_dist.
            # Heuristic: if measured_dist is very close to max range, treat as max range.
            is_max_range_reading = (measured_dist >= self.max_sensor_range_pixels - self.cell_size_pixels / 4.0)

            trace_length_for_free = self.max_sensor_range_pixels if is_max_range_reading else measured_dist

            # Trace along the beam to mark cells as FREE
            # Step by a fraction of cell size for better coverage.
            ray_tracing_step = self.cell_size_pixels / 2.0  # Step half a cell

            current_dist_on_ray = ray_tracing_step  # Start a bit away from the robot's center
            while current_dist_on_ray < trace_length_for_free - (
                    self.cell_size_pixels / 4.0):  # Stop before the perceived obstacle
                # Calculate point on the ray
                px = robot_x + current_dist_on_ray * math.cos(global_beam_angle)
                py = robot_y + current_dist_on_ray * math.sin(global_beam_angle)

                # Convert world (pixel) coordinates to grid cell indices
                grid_col_idx = int(px / self.cell_size_pixels)  # j
                grid_row_idx = int(py / self.cell_size_pixels)  # i

                if 0 <= grid_row_idx < self.height and 0 <= grid_col_idx < self.width:
                    self.grid[grid_row_idx, grid_col_idx] += self.log_odds_free

                current_dist_on_ray += ray_tracing_step
                # Safety break if somehow current_dist_on_ray exceeds max sensor range significantly
                if current_dist_on_ray > self.max_sensor_range_pixels + ray_tracing_step:
                    break

            # Mark the cell at the END of the beam as OCCUPIED,
            # but only if it's NOT a max-range reading (i.e., an obstacle was detected).
            if not is_max_range_reading and measured_dist < self.max_sensor_range_pixels:
                end_point_x = robot_x + measured_dist * math.cos(global_beam_angle)
                end_point_y = robot_y + measured_dist * math.sin(global_beam_angle)

                end_grid_col_idx = int(end_point_x / self.cell_size_pixels)
                end_grid_row_idx = int(end_point_y / self.cell_size_pixels)

                if 0 <= end_grid_row_idx < self.height and 0 <= end_grid_col_idx < self.width:
                    self.grid[end_grid_row_idx, end_grid_col_idx] += self.log_odds_occupied
        # --- End Iterative Ray-Casting ---

        # Clamp log-odds values to prevent them from becoming too large or too small
        np.clip(self.grid, self.log_odds_min_val, self.log_odds_max_val, out=self.grid)

    def get_probability_grid(self):
        # Convert log-odds to probabilities: P = 1 - 1 / (1 + exp(log_odds))
        # or P = exp(log_odds) / (1 + exp(log_odds))
        exp_log_odds = np.exp(self.grid)
        prob_grid = exp_log_odds / (1 + exp_log_odds)
        return prob_grid

    def get_grayscale_grid(self):
        prob_grid = self.get_probability_grid()
        # Invert probability for grayscale: occupied (high prob) = dark, free (low prob) = light
        grayscale = (1 - prob_grid) * 255
        return grayscale.astype(np.uint8)