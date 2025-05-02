from dataclasses import dataclass

import numpy as np

from maps import Maps

@dataclass
class Config:
    """
    Configuration class for the application.
    Allows specifying desired grid dimensions while keeping window size fixed.
    """

    # --- User Definable Grid Size ---
    # Change these values to set the number of cells in the grid
    GRID_WIDTH = 50
    GRID_HEIGHT = 50

    # --- Fixed Window Size ---
    # Define the total window size in pixels. This will remain constant.
    FIXED_WINDOW_WIDTH = 1000  # Pixels
    FIXED_WINDOW_HEIGHT = 1000  # Pixels

    # --- Calculated Cell Size ---
    # Calculate cell size dynamically to fit the desired grid into the fixed window
    # We take the minimum scale factor to ensure it fits in both dimensions.
    cell_width_pixels = FIXED_WINDOW_WIDTH // GRID_WIDTH
    cell_height_pixels = FIXED_WINDOW_HEIGHT // GRID_HEIGHT
    CELL_SIZE = min(cell_width_pixels, cell_height_pixels)
    # Ensure the cell size is at least 1 pixel
    CELL_SIZE = max(1, CELL_SIZE)

    # --- Maze and Grid Setup ---
    # Generate maze using the desired grid dimensions
    # NOTE: Maps.generate_maze needs height first, then width
    maze_grid = Maps.generate_maze(GRID_HEIGHT, GRID_WIDTH, complexity=0)

    # --- Landmarks and Start Position ---
    # These functions now use the dynamically calculated CELL_SIZE
    landmarks = Maps.find_corner_landmarks(maze_grid, CELL_SIZE)
    # landmarks = Maps.find_random_landmarks(maze_grid, CELL_SIZE, num_landmarks=30) # Alternative
    start_pos = Maps.find_empty_spot(maze_grid, CELL_SIZE)

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
    ORANGE = (255, 165, 0)
    DARK_BLUE = (255, 87, 51)
    PURPLE = (128, 0, 128)