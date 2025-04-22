import numpy as np
import random

class Maps:

    @staticmethod
    def generate_bordered_map(width: int, height: int) -> np.ndarray:
        grid = np.zeros((height, width), dtype=int)
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        return grid

    @staticmethod
    def generate_building_map(width: int, height: int) -> np.ndarray:
        grid = np.ones((height, width), dtype=int)

        room_w, room_h = 6, 4
        padding = 2
        room_centers = []

        for i in range(padding, height - room_h, room_h + padding):
            for j in range(padding, width - room_w, room_w + padding):
                if random.random() < 0.8:
                    grid[i:i+room_h, j:j+room_w] = 0
                    cx, cy = i + room_h // 2, j + room_w // 2
                    room_centers.append((cx, cy))

        # Connect each room to the next with corridors
        for idx, (cx, cy) in enumerate(room_centers):
            if idx + 1 < len(room_centers):
                nx, ny = room_centers[idx + 1]
                if random.random() < 0.5:
                    # Horizontal then vertical
                    grid[cx, min(cy, ny):max(cy, ny)+1] = 0
                    grid[min(cx, nx):max(cx, nx)+1, ny] = 0
                else:
                    # Vertical then horizontal
                    grid[min(cx, nx):max(cx, nx)+1, cy] = 0
                    grid[nx, min(cy, ny):max(cy, ny)+1] = 0

        grid[1][0] = 0
        grid[height - 2][width - 1] = 0
        return grid

    @staticmethod
    def generate_maze(width: int, height: int) -> np.ndarray:
        # Ensure odd dimensions
        width = width if width % 2 == 1 else width - 1
        height = height if height % 2 == 1 else height - 1

        # Initialize with walls
        maze = np.ones((height, width), dtype=int)

        def carve_passages(x: int, y: int):
            directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 1 <= nx < height - 1 and 1 <= ny < width - 1:
                    if maze[nx][ny] == 1:
                        maze[nx][ny] = 0
                        maze[x + dx // 2][y + dy // 2] = 0
                        carve_passages(nx, ny)

        # Start carving from a random odd cell
        start_x, start_y = 1, 1
        maze[start_x][start_y] = 0
        carve_passages(start_x, start_y)

        # Entrance and exit
        maze[1][0] = 0
        maze[height - 2][width - 1] = 0

        return maze

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
