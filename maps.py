import numpy as np
import random

class Maps:

    @staticmethod
    def find_empty_spot(grid, cell_size=None):
        """ Find a random empty spot (cell with value 0) in the grid and return its coordinates. """
        if not grid.any() or not grid[0].any():
            return None

        rows, cols = grid.shape
        empty_cells = []

        # Find all empty cells
        for y in range(rows):
            for x in range(cols):
                if grid[y][x] == 0:
                    empty_cells.append((x, y))

        if not empty_cells:
            return None  # No empty cells found

        # Choose a random empty cell
        grid_x, grid_y = random.choice(empty_cells)

        # Convert to Cartesian coordinates if cell_size is provided
        if cell_size is not None:
            # Add random offset within the cell to avoid placing everything at cell corners
            offset_x = 0.5 * cell_size
            offset_y = 0.5 * cell_size
            return (grid_x * cell_size) + offset_x, (grid_y * cell_size) + offset_y

        # Otherwise return grid coordinates
        return grid_x, grid_y

    @staticmethod
    def generate_bordered_map(width: int, height: int) -> np.ndarray:
        grid = np.zeros((height, width), dtype=int)
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        return grid

    @staticmethod
    def generate_maze(width: int, height: int, complexity: float = 1.0):
        """
        Generates a maze as a 2D list (bitmap) of 0s (paths) and 1s (walls).

        Args:
            width: The width of the maze grid. Recommended >= 3.
            height: The height of the maze grid. Recommended >= 3.
            complexity: A float between 0.0 and 1.0.
                        1.0 creates a dense, complex maze structure.
                        0.0 creates a mostly open area with few obstacles.
                        Values in between interpolate the density.

        Returns:
            A list of lists representing the maze bitmap (0=path, 1=wall).
            Returns an empty list if the width or height are too small.
        """
        if width < 3 or height < 3:
            print("Warning: Maze dimensions should be at least 3x3 for meaningful generation.")
            # Return a simple grid for very small sizes or handle as an error
            if complexity > 0.5:
                return [[1] * width for _ in range(height)]  # Mostly walls
            else:
                return [[0] * width for _ in range(height)]  # Mostly empty

        # Clamp complexity
        complexity = max(0.0, min(1.0, complexity))

        # Initialize grid: 1 = Wall, 0 = Path
        # Start with a grid full of walls
        maze = [[1] * width for _ in range(height)]

        # --- Recursive Backtracker (DFS) for base maze generation ---
        # Choose a starting cell (must be odd coordinates for traditional carving)
        # For simplicity here, we'll just start at (0,0) and ensure it's a path
        start_x, start_y = 0, 0
        maze[start_y][start_x] = 0
        stack = [(start_x, start_y)]
        visited_carving = {(start_x, start_y)}  # Keep track of visited cells during carving

        while stack:
            cx, cy = stack[-1]  # Current cell

            # Find unvisited neighbors (2 steps away)
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                # Check bounds and if the neighbor cell is currently a wall (effectively unvisited)
                if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 1:
                    neighbors.append((nx, ny))

            if neighbors:
                # Choose a random neighbor
                nx, ny = random.choice(neighbors)

                # Carve path to the neighbor
                # Mark neighbor as path
                maze[ny][nx] = 0
                visited_carving.add((nx, ny))

                # Carve the wall between current cell and neighbor
                wall_x, wall_y = cx + (dx // 2), cy + (dy // 2)  # Get the wall cell coordinates
                # Find the wall coordinates based on direction chosen
                if nx == cx + 2:
                    wall_x = cx + 1; wall_y = cy
                elif nx == cx - 2:
                    wall_x = cx - 1; wall_y = cy
                elif ny == cy + 2:
                    wall_x = cx; wall_y = cy + 1
                elif ny == cy - 2:
                    wall_x = cx; wall_y = cy - 1

                if 0 <= wall_x < width and 0 <= wall_y < height:  # Ensure wall is within bounds
                    maze[wall_y][wall_x] = 0

                # Move to the neighbor
                stack.append((nx, ny))
            else:
                # Backtrack
                stack.pop()

        # --- Complexity Adjustment: Remove Walls ---
        # Find potential internal walls to remove (walls separating two paths)
        removable_walls = []
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if maze[r][c] == 1:  # If it's a wall
                    # Check if it's between two path cells (horizontally or vertically)
                    is_horizontal_divider = (maze[r][c - 1] == 0 and maze[r][c + 1] == 0)
                    is_vertical_divider = (maze[r - 1][c] == 0 and maze[r + 1][c] == 0)
                    if is_horizontal_divider or is_vertical_divider:
                        removable_walls.append((c, r))  # Store as (x, y) or (col, row)

        # Calculate the number of walls to remove based on INVERSE complexity
        # (1.0 - complexity) gives proportion to remove. 0.0 complexity removes max, 1.0 removes 0.
        num_walls_to_remove = int(len(removable_walls) * (1.0 - complexity))

        # Shuffle the list of removable walls
        random.shuffle(removable_walls)

        # Remove the walls
        for i in range(num_walls_to_remove):
            if removable_walls:  # Check if list is not empty
                wall_x, wall_y = removable_walls.pop()
                maze[wall_y][wall_x] = 0  # Turn wall into path

        return np.array(maze)

    @staticmethod
    def find_random_landmarks(grid, cell_size, num_landmarks=50, min_distance=None):
        """
        Randomly place landmarks within the empty spaces (cells with value 0) of the grid
        and return their Cartesian coordinates.
        """
        if not grid.any() or not grid[0].any():
            return []

        rows = len(grid)
        cols = len(grid[0])

        # Find all empty cells
        empty_cells = []
        for y in range(rows):
            for x in range(cols):
                if grid[y][x] == 0:
                    empty_cells.append((x, y))

        if not empty_cells:
            return []  # No empty cells to place landmarks

        # Function to calculate Cartesian coordinates from grid position
        def grid_to_cartesian(grid_x, grid_y):
            # Random position within the cell (to avoid all landmarks being at cell corners)
            offset_x = random.random() * cell_size
            offset_y = random.random() * cell_size
            cart_x = (grid_x * cell_size) + offset_x
            cart_y = (grid_y * cell_size) + offset_y
            return cart_x, cart_y

        # Function to calculate Euclidean distance between two points
        def distance(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        landmarks = []
        attempts = 0
        max_attempts = len(empty_cells) * 10  # Prevent infinite loops

        # Keep trying to add landmarks until we have enough or run out of attempts
        while len(landmarks) < num_landmarks and attempts < max_attempts:
            attempts += 1

            # If we've tried too many times and have some landmarks, return what we have
            if attempts > max_attempts // 2 and landmarks:
                break

            # Choose a random empty cell
            grid_x, grid_y = random.choice(empty_cells)
            new_landmark = grid_to_cartesian(grid_x, grid_y)

            # Check minimum distance if specified
            if min_distance is not None:
                too_close = False
                for existing_landmark in landmarks:
                    if distance(new_landmark, existing_landmark) < min_distance:
                        too_close = True
                        break

                if too_close:
                    continue  # Try again with a different cell

            landmarks.append(new_landmark)

        return landmarks

    @staticmethod
    def find_corner_landmarks(grid, cell_size):
        """
        Find corners in a binary grid where 1 represents an obstacle, and calculate
        their Cartesian coordinates.
        """
        if not grid.any():
            return []

        rows = len(grid)
        cols = len(grid[0])
        corners = []

        # Define the 8 adjacent positions (clockwise from top)
        # N, NE, E, SE, S, SW, W, NW
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

        # Define corner patterns - these are the patterns that indicate corners
        # Each pattern is a list of four adjacent cells that form a 2x2 grid
        # Format: [top-left, top-right, bottom-left, bottom-right]
        corner_patterns = {
            # NW corner: 1 with 0s on its right and below
            "NW": [1, 0, 0, 0],
            # NE corner: 1 with 0s on its left and below
            "NE": [0, 1, 0, 0],
            # SW corner: 1 with 0s on its right and above
            "SW": [0, 0, 1, 0],
            # SE corner: 1 with 0s on its left and above
            "SE": [0, 0, 0, 1]
        }

        # Corner offsets within a cell based on corner type
        corner_offsets = {
            "NW": (0, 0),  # Top-left corner
            "NE": (cell_size, 0),  # Top-right corner
            "SW": (0, cell_size),  # Bottom-left corner
            "SE": (cell_size, cell_size)  # Bottom-right corner
        }

        # Check each cell that could potentially be part of a 2x2 grid
        for y in range(rows - 1):
            for x in range(cols - 1):
                # Get the values of the 2x2 grid
                grid_2x2 = [
                    grid[y][x],  # top-left
                    grid[y][x + 1],  # top-right
                    grid[y + 1][x],  # bottom-left
                    grid[y + 1][x + 1]  # bottom-right
                ]

                # Check for each corner pattern
                for corner_type, pattern in corner_patterns.items():
                    # A corner exists when the 2x2 grid matches the pattern
                    # For example, a NW corner exists when the top-left is 1 and the rest are 0
                    matches = True
                    for i in range(4):
                        if grid_2x2[i] != pattern[i]:
                            matches = False
                            break

                    if matches:
                        # Calculate the base coordinates for this 2x2 grid
                        base_x = x * cell_size
                        base_y = y * cell_size

                        # Get the offset for this corner type
                        offset_x, offset_y = corner_offsets[corner_type]

                        # Add the corner to the list
                        corners.append((base_x + offset_x, base_y + offset_y))

        # Edge cases - check cells at the boundaries of the grid

        # Top edge (except corners)
        for x in range(1, cols - 1):
            if grid[0][x] == 1 and grid[0][x - 1] == 0:  # Left edge of obstacle
                corners.append((x * cell_size, 0))
            if grid[0][x] == 1 and grid[0][x + 1] == 0:  # Right edge of obstacle
                corners.append(((x + 1) * cell_size, 0))

        # Bottom edge (except corners)
        for x in range(1, cols - 1):
            if grid[rows - 1][x] == 1 and grid[rows - 1][x - 1] == 0:  # Left edge of obstacle
                corners.append((x * cell_size, rows * cell_size))
            if grid[rows - 1][x] == 1 and grid[rows - 1][x + 1] == 0:  # Right edge of obstacle
                corners.append(((x + 1) * cell_size, rows * cell_size))

        # Left edge (except corners)
        for y in range(1, rows - 1):
            if grid[y][0] == 1 and grid[y - 1][0] == 0:  # Top edge of obstacle
                corners.append((0, y * cell_size))
            if grid[y][0] == 1 and grid[y + 1][0] == 0:  # Bottom edge of obstacle
                corners.append((0, (y + 1) * cell_size))

        # Right edge (except corners)
        for y in range(1, rows - 1):
            if grid[y][cols - 1] == 1 and grid[y - 1][cols - 1] == 0:  # Top edge of obstacle
                corners.append((cols * cell_size, y * cell_size))
            if grid[y][cols - 1] == 1 and grid[y + 1][cols - 1] == 0:  # Bottom edge of obstacle
                corners.append((cols * cell_size, (y + 1) * cell_size))

        # Corner cases of the grid
        if grid[0][0] == 1:
            corners.append((0, 0))  # Top-left corner of the grid
        if grid[0][cols - 1] == 1:
            corners.append((cols * cell_size, 0))  # Top-right corner of the grid
        if grid[rows - 1][0] == 1:
            corners.append((0, rows * cell_size))  # Bottom-left corner of the grid
        if grid[rows - 1][cols - 1] == 1:
            corners.append((cols * cell_size, rows * cell_size))  # Bottom-right corner of the grid

        # Interior corners
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if grid[y][x] == 0:  # Only check empty cells
                    # Check for corners formed by obstacle configurations
                    # NW corner
                    if grid[y - 1][x] == 1 and grid[y][x - 1] == 1:
                        corners.append((x * cell_size, y * cell_size))
                    # NE corner
                    if grid[y - 1][x] == 1 and grid[y][x + 1] == 1:
                        corners.append(((x + 1) * cell_size, y * cell_size))
                    # SW corner
                    if grid[y + 1][x] == 1 and grid[y][x - 1] == 1:
                        corners.append((x * cell_size, (y + 1) * cell_size))
                    # SE corner
                    if grid[y + 1][x] == 1 and grid[y][x + 1] == 1:
                        corners.append(((x + 1) * cell_size, (y + 1) * cell_size))

        # Remove duplicate corners
        return list(set(corners))
