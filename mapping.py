import numpy as np
import math
from config import Config
import scipy.linalg


class OccupancyGrid:
    def __init__(self, robot):
        self.robot = robot
        self.width = Config.GRID_WIDTH
        self.height = Config.GRID_HEIGHT
        # Initialize a grid with unknown probability (0.5)
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)

        self.log_odds_occupied = np.log(0.9 / 0.1)
        self.log_odds_free = np.log(0.3 / 0.7)

        self.alpha = Config.CELL_SIZE
        self.beta = 5.0 * np.pi / 180.0  # 5 degrees in radians
        # maximum sensor range in meters
        self.z_max_m = robot.sensor_range

        xs = np.arange(self.width) * Config.CELL_SIZE  # pixel x-coordinates
        ys = np.arange(self.height) * Config.CELL_SIZE  # pixel y-coordinates
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")  # shape: (H, W)
        self.grid_position_m = np.stack((grid_x, grid_y))  # shape: (2, H, W)

    def update(self, pose, z):

        dx = self.grid_position_m.copy()
        dx[0] = dx[0] - pose[0]
        dx[1] = dx[1] - pose[1]
        theta_to_grid = np.arctan2(dx[1, :, :], dx[0, :, :]) - pose[2]  # matrix of all bearings from robot to cell

        # Wrap to +pi / - pi
        theta_to_grid[theta_to_grid > np.pi] -= 2. * np.pi
        theta_to_grid[theta_to_grid < -np.pi] += 2. * np.pi

        dist_to_grid = scipy.linalg.norm(dx, axis=0)  # matrix of L2 distance to all cells from robot

        # For each laser beam
        for z_i in z:
            r = z_i[0]
            b = z_i[1]
            # reading already in meters
            r_m = r
            # skip beams beyond sensor range

            if r_m <= self.alpha:
                continue
            if r_m > self.z_max_m:
                continue

            # ray‐cast along beam in robot frame
            # compute global beam angle
            beam_angle = pose[2] + b
            # limit free‑space to either the hit distance or max range
            max_dist = min(r_m, self.z_max_m)
            # number of cells to traverse
            steps = int(max_dist / Config.CELL_SIZE)
            for step in range(1, steps):
                # point along the ray
                px = pose[0] + step * Config.CELL_SIZE * math.cos(beam_angle)
                py = pose[1] + step * Config.CELL_SIZE * math.sin(beam_angle)
                # convert to grid indices
                j = int(px / Config.CELL_SIZE)
                i = int(py / Config.CELL_SIZE)
                if 0 <= i < self.height and 0 <= j < self.width:
                    self.grid[i, j] += self.log_odds_free

            # if this beam hit an obstacle (not max‑range), mark the endpoint occupied
            if r_m < self.z_max_m - (self.alpha / 2):
                ex = pose[0] + r_m * math.cos(beam_angle)
                ey = pose[1] + r_m * math.sin(beam_angle)
                ej = int(ex / Config.CELL_SIZE)
                ei = int(ey / Config.CELL_SIZE)
                if 0 <= ei < self.height and 0 <= ej < self.width:
                    self.grid[ei, ej] += self.log_odds_occupied

        np.clip(self.grid, -10, 10, out=self.grid)

    def get_grayscale_grid(self):
        prob_grid = 1 / (1 + np.exp(-self.grid))
        grayscale = (1 - prob_grid) * 255
        return grayscale.astype(np.uint8)
