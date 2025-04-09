from dataclasses import dataclass
from maps import *

@dataclass
class Config:
    """
    Configuration class for the application.
    """
    maze_grid = generate_bordered_map(23,23)
    GRID_HEIGHT, GRID_WIDTH = maze_grid.shape
    CELL_SIZE = 20  # pixels per cell

    # Pygame window size
    WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
    WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

    # Define colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (120, 120, 120)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    NEON_PINK = (255, 20, 147)
    AQUA = (0, 255, 255)