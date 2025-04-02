import numpy as np
import math
from config import Config

class OccupancyGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Initialize grid with unknown probability (0.5)
        self.grid = 0.5 * np.ones((height, width))

    def update(self, robot, sensor_readings):
        x, y, theta = robot.get_pose()
        for i, distance in enumerate(sensor_readings):
            sensor_angle = theta + robot.sensor_angles[i]
            # Mark cells along the ray as free
            for d in range(0, int(distance)):
                cell_x = int((x + d * math.cos(sensor_angle)) // Config.CELL_SIZE)
                cell_y = int((y + d * math.sin(sensor_angle)) // Config.CELL_SIZE)
                if 0 <= cell_x < self.width and 0 <= cell_y < self.height:
                    self.grid[cell_y, cell_x] = 0.0  # free space
            # Mark the cell where obstacle is detected as occupied
            cell_x = int((x + distance * math.cos(sensor_angle)) // Config.CELL_SIZE)
            cell_y = int((y + distance * math.sin(sensor_angle)) // Config.CELL_SIZE)
            if 0 <= cell_x < self.width and 0 <= cell_y < self.height:
                self.grid[cell_y, cell_x] = 1.0  # occupied

    def get_frontiers(self):
        """
        Identify frontier cells (free cells adjacent to unknown cells).
        """
        frontiers = []
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                if self.grid[i, j] == 0.0:  # free cell
                    # Check surrounding 3x3 neighborhood for unknown cells (0.5)
                    neighbors = self.grid[i-1:i+2, j-1:j+2]
                    if np.any(neighbors == 0.5):
                        frontiers.append((j, i))
        return frontiers