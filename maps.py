import numpy as np
import random

class Room:
    """ Represents a rectangular room on the grid. """

    def __init__(self, x, y, w, h):
        self.x1 = x  # Top-left x-coordinate
        self.y1 = y  # Top-left y-coordinate
        self.w = w  # Width
        self.h = h  # Height
        self.x2 = x + w  # Bottom-right x-coordinate (exclusive)
        self.y2 = y + h  # Bottom-right y-coordinate (exclusive)
        self.center_x = (self.x1 + self.x2) // 2
        self.center_y = (self.y1 + self.y2) // 2

    def intersects(self, other_room, buffer=1):
        """ Checks if this room intersects with another room, including a buffer zone. """
        # Check if the rectangles (plus buffer) overlap
        return (self.x1 < other_room.x2 + buffer and
                self.x2 + buffer > other_room.x1 and
                self.y1 < other_room.y2 + buffer and
                self.y2 + buffer > other_room.y1)


class Maps:
    # Constants
    WALL = 1  # Represents a wall cell
    FLOOR = 0  # Represents an empty floor/corridor cell

    @staticmethod
    def create_house(grid_width, grid_height, num_rooms, min_room_size=4, max_room_size=10, max_placement_attempts=1000, corridor_width=1):
        """
        Generates a 2D list representing a map with rooms and corridors.

        Args:
            grid_width (int): The width of the map grid.
            grid_height (int): The height of the map grid.
            num_rooms (int): The target number of rooms to place.
            min_room_size (int): The minimum width/height of a room.
            max_room_size (int): The maximum width/height of a room.
            max_placement_attempts (int): Max attempts to place rooms overall.

        Returns:
            np.ndarray: A 2D array (bitmap) where 1 represents a wall
                         and 0 represents floor/empty space.
        """
        if not (grid_width > 0 and grid_height > 0 and num_rooms >= 0):
            raise ValueError("Grid dimensions must be positive, num_rooms non-negative.")
        if min_room_size > max_room_size:
            raise ValueError("min_room_size cannot be greater than max_room_size.")
        if max_room_size >= min(grid_width, grid_height) - 2:
            print(f"Warning: max_room_size ({max_room_size}) is large relative to grid dimensions "
                 f"({grid_width}x{grid_height}). Room placement might be difficult or fail.")

        # 1. Initialize grid full of walls
        grid = [[Maps.WALL for _ in range(grid_width)] for _ in range(grid_height)]

        if num_rooms == 0:
            return np.array(grid)  # Return empty grid if no rooms requested

        placed_rooms = []
        attempts = 0

        # 2. Try placing rooms
        while len(placed_rooms) < num_rooms and attempts < max_placement_attempts:
            attempts += 1

            # Generate random room dimensions
            room_w = random.randint(min_room_size, max_room_size)
            room_h = random.randint(min_room_size, max_room_size)

            # Generate random position (ensure room stays within grid boundaries + 1 cell border)
            max_x = grid_width - room_w - 1
            max_y = grid_height - room_h - 1

            if max_x <= 1 or max_y <= 1:  # Check if grid is too small for any room placement with border
                if len(placed_rooms) == 0:  # Only raise error if we couldn't place *any* room
                    raise ValueError(f"Grid {grid_width}x{grid_height} too small for "
                                    f"room size {min_room_size}-{max_room_size} with border.")
                else:  # Otherwise, just stop trying if space runs out
                    print("Warning: Grid space potentially exhausted for new rooms.")
                    break

            room_x = random.randint(1, max_x)
            room_y = random.randint(1, max_y)

            # Create a potential new room object
            new_room = Room(room_x, room_y, room_w, room_h)

            # 3. Check for collisions with existing rooms (use a buffer)
            collision = False
            for existing_room in placed_rooms:
                if new_room.intersects(existing_room, buffer=1):
                    collision = True
                    break

            # 4. If no collision, place the room and connect it
            if not collision:
                Maps._create_room_on_grid(grid, new_room)

                # 5. Connect to the previous room (if applicable)
                if placed_rooms:  # If this is not the first room
                    prev_room = placed_rooms[-1]
                    Maps._connect_rooms(grid, new_room, prev_room, corridor_width)

                placed_rooms.append(new_room)

        if len(placed_rooms) < num_rooms:
            print(f"Warning: Only placed {len(placed_rooms)} out of {num_rooms} desired rooms "
                 f"after {max_placement_attempts} attempts.")

        return np.array(grid)

    @staticmethod
    def _create_room_on_grid(grid, room):
        """ Carves a room (sets cells to FLOOR) onto the grid. """
        grid_height = len(grid)
        grid_width = len(grid[0])
        # Iterate within the room boundaries (exclusive of x2, y2)
        for y in range(room.y1, room.y2):
            for x in range(room.x1, room.x2):
                # Basic bounds check (should be guaranteed by placement logic but safe)
                if 0 <= y < grid_height and 0 <= x < grid_width:
                    grid[y][x] = Maps.FLOOR

    @staticmethod
    def _create_h_corridor(grid, x1, x2, y, width):
        """ Carves a horizontal corridor with specified width. """
        grid_height = len(grid)
        grid_width = len(grid[0])
        half_width = width // 2

        for y_offset in range(-half_width, half_width + 1):
            current_y = y + y_offset
            if 0 <= current_y < grid_height:
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    if 0 <= x < grid_width:
                        grid[current_y][x] = Maps.FLOOR

    @staticmethod
    def _create_v_corridor(grid, y1, y2, x, width):
        """ Carves a vertical corridor with specified width. """
        grid_height = len(grid)
        grid_width = len(grid[0])
        half_width = width // 2

        for x_offset in range(-half_width, half_width + 1):
            current_x = x + x_offset
            if 0 <= current_x < grid_width:
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    if 0 <= y < grid_height:
                        grid[y][current_x] = Maps.FLOOR

    @staticmethod
    def _connect_rooms(grid, room1, room2, corridor_width):
        """ Connects the centers of two rooms with an L-shaped corridor. """
        center1_x, center1_y = room1.center_x, room1.center_y
        center2_x, center2_y = room2.center_x, room2.center_y

        # Randomly decide the turn direction
        if random.random() < 0.5:
            # Horizontal first
            Maps._create_h_corridor(grid, center1_x, center2_x, center1_y, corridor_width)
            Maps._create_v_corridor(grid, center1_y, center2_y, center2_x, corridor_width)
        else:
            # Vertical first
            Maps._create_v_corridor(grid, center1_y, center2_y, center1_x, corridor_width)
            Maps._create_h_corridor(grid, center1_x, center2_x, center2_y, corridor_width)


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
        Generates a maze as a 2D numpy array of 0s (paths) and 1s (walls).
        Starts with a bordered map and adds obstacles inside based on complexity.

        Args:
            width: The width of the maze grid. Recommended >= 3.
            height: The height of the maze grid. Recommended >= 3.
            complexity: A float between 0.0 and 1.0.
                        1.0 creates a dense, complex maze structure.
                        0.0 creates a mostly open area with just the border.
                        Values in between interpolate the density.

        Returns:
            A numpy array representing the maze bitmap (0=path, 1=wall).
        """
        if width < 3 or height < 3:
            print("Warning: Maze dimensions should be at least 3x3 for meaningful generation.")
            # Return a simple grid for very small sizes or handle as an error
            if complexity > 0.5:
                return np.ones((height, width), dtype=int)  # Mostly walls
            else:
                return np.zeros((height, width), dtype=int)  # Mostly empty

        # Clamp complexity
        complexity = max(0.0, min(1.0, complexity))

        # Start with a bordered map (outer walls only)
        maze = Maps.generate_bordered_map(width, height)

        # If complexity is 0, return just the bordered map
        if complexity == 0:
            return maze

        # Find all potential internal wall locations (not on the border)
        potential_walls = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                potential_walls.append((x, y))

        # Calculate how many walls to add based on complexity
        # Leave some empty spaces to ensure the maze is traversable
        max_walls = len(potential_walls) * 0.7  # Cap at 70% of interior filled
        num_walls_to_add = int(max_walls * complexity)

        # Shuffle the potential wall locations
        random.shuffle(potential_walls)

        # Add walls
        for i in range(num_walls_to_add):
            if potential_walls:  # Check if list is not empty
                wall_x, wall_y = potential_walls.pop()
                maze[wall_y, wall_x] = 1

        # Ensure there's a connected path through the maze
        # Simple algorithm: ensure at least one path from top to bottom and left to right
        # This keeps the maze traversable even at high complexity
        if complexity > 0.5:
            # Create at least one horizontal and one vertical path
            path_y = random.randint(1, height - 2)
            path_x = random.randint(1, width - 2)

            # Clear horizontal path
            for x in range(1, width - 1):
                maze[path_y, x] = 0

            # Clear vertical path
            for y in range(1, height - 1):
                maze[y, path_x] = 0

        return maze

    @staticmethod
    def add_noise_to_maze(maze, swap_probability):
        """
        Adds random noise to a maze by swapping walls (1) with paths (0) based on a probability.

        Args:
            maze: 2D numpy array representing the maze (0=path, 1=wall)
            swap_probability: Float between 0 and 1, the probability of swapping a cell's value
        """
        # Get maze dimensions
        height, width = maze.shape

        # Calculate how many swaps to perform based on probability
        total_cells = height * width
        expected_swaps = int(total_cells * swap_probability)

        # Perform the expected number of swaps
        for _ in range(expected_swaps):
            # Choose random interior coordinates (not on the border)
            y = random.randint(1, height - 2)
            x = random.randint(1, width - 2)

            # Only proceed if this is a wall
            if maze[y, x] == 1:
                # Find neighboring path cells
                neighbors = []
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and maze[ny, nx] == 0:
                        neighbors.append((ny, nx))

                # If there are path neighbors, swap with one
                if neighbors:
                    ny, nx = random.choice(neighbors)
                    maze[y, x] = 0  # Wall becomes path
                    maze[ny, nx] = 1  # Path becomes wall

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
        their Cartesian coordinates. Only add a corner if it is adjacent to a wall cell.
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

        # Helper function to check if a Cartesian coordinate is adjacent to a wall cell
        def has_wall_neighbor_fn(corner_x, corner_y):
            grid_x = int(corner_x / cell_size)
            grid_y = int(corner_y / cell_size)
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny = grid_y + dy
                    nx = grid_x + dx
                    if 0 <= ny < rows and 0 <= nx < cols and grid[ny][nx] == 1:
                        return True
            return False

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

                        # Only add if adjacent to a wall cell
                        corner_x = base_x + offset_x
                        corner_y = base_y + offset_y
                        if has_wall_neighbor_fn(corner_x, corner_y):
                            corners.append((corner_x, corner_y))

        # Edge cases - check cells at the boundaries of the grid

        # Top edge (except corners)
        for x in range(1, cols - 1):
            if grid[0][x] == 1 and grid[0][x - 1] == 0:
                corner_x = x * cell_size
                corner_y = 0
                if has_wall_neighbor_fn(corner_x, corner_y):
                    corners.append((corner_x, corner_y))
            if grid[0][x] == 1 and grid[0][x + 1] == 0:
                corner_x = (x + 1) * cell_size
                corner_y = 0
                if has_wall_neighbor_fn(corner_x, corner_y):
                    corners.append((corner_x, corner_y))

        # Bottom edge (except corners)
        for x in range(1, cols - 1):
            if grid[rows - 1][x] == 1 and grid[rows - 1][x - 1] == 0:
                corner_x = x * cell_size
                corner_y = rows * cell_size
                if has_wall_neighbor_fn(corner_x, corner_y):
                    corners.append((corner_x, corner_y))
            if grid[rows - 1][x] == 1 and grid[rows - 1][x + 1] == 0:
                corner_x = (x + 1) * cell_size
                corner_y = rows * cell_size
                if has_wall_neighbor_fn(corner_x, corner_y):
                    corners.append((corner_x, corner_y))

        # Left edge (except corners)
        for y in range(1, rows - 1):
            if grid[y][0] == 1 and grid[y - 1][0] == 0:
                corner_x = 0
                corner_y = y * cell_size
                if has_wall_neighbor_fn(corner_x, corner_y):
                    corners.append((corner_x, corner_y))
            if grid[y][0] == 1 and grid[y + 1][0] == 0:
                corner_x = 0
                corner_y = (y + 1) * cell_size
                if has_wall_neighbor_fn(corner_x, corner_y):
                    corners.append((corner_x, corner_y))

        # Right edge (except corners)
        for y in range(1, rows - 1):
            if grid[y][cols - 1] == 1 and grid[y - 1][cols - 1] == 0:
                corner_x = cols * cell_size
                corner_y = y * cell_size
                if has_wall_neighbor_fn(corner_x, corner_y):
                    corners.append((corner_x, corner_y))
            if grid[y][cols - 1] == 1 and grid[y + 1][cols - 1] == 0:
                corner_x = cols * cell_size
                corner_y = (y + 1) * cell_size
                if has_wall_neighbor_fn(corner_x, corner_y):
                    corners.append((corner_x, corner_y))

        # Corner cases of the grid
        if grid[0][0] == 1:
            corner_x = 0
            corner_y = 0
            if has_wall_neighbor_fn(corner_x, corner_y):
                corners.append((corner_x, corner_y))  # Top-left corner of the grid
        if grid[0][cols - 1] == 1:
            corner_x = cols * cell_size
            corner_y = 0
            if has_wall_neighbor_fn(corner_x, corner_y):
                corners.append((corner_x, corner_y))  # Top-right corner of the grid
        if grid[rows - 1][0] == 1:
            corner_x = 0
            corner_y = rows * cell_size
            if has_wall_neighbor_fn(corner_x, corner_y):
                corners.append((corner_x, corner_y))  # Bottom-left corner of the grid
        if grid[rows - 1][cols - 1] == 1:
            corner_x = cols * cell_size
            corner_y = rows * cell_size
            if has_wall_neighbor_fn(corner_x, corner_y):
                corners.append((corner_x, corner_y))  # Bottom-right corner of the grid

        # Interior corners
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if grid[y][x] == 0:
                    # NW corner
                    if grid[y - 1][x] == 1 and grid[y][x - 1] == 1:
                        corner_x = x * cell_size
                        corner_y = y * cell_size
                        if has_wall_neighbor_fn(corner_x, corner_y):
                            corners.append((corner_x, corner_y))
                    # NE corner
                    if grid[y - 1][x] == 1 and grid[y][x + 1] == 1:
                        corner_x = (x + 1) * cell_size
                        corner_y = y * cell_size
                        if has_wall_neighbor_fn(corner_x, corner_y):
                            corners.append((corner_x, corner_y))
                    # SW corner
                    if grid[y + 1][x] == 1 and grid[y][x - 1] == 1:
                        corner_x = x * cell_size
                        corner_y = (y + 1) * cell_size
                        if has_wall_neighbor_fn(corner_x, corner_y):
                            corners.append((corner_x, corner_y))
                    # SE corner
                    if grid[y + 1][x] == 1 and grid[y][x + 1] == 1:
                        corner_x = (x + 1) * cell_size
                        corner_y = (y + 1) * cell_size
                        if has_wall_neighbor_fn(corner_x, corner_y):
                            corners.append((corner_x, corner_y))

        # Remove duplicate corners
        return list(set(corners))

    @staticmethod
    def generate_static_dust(height, width, grid, cell_size, color):
        """
        Generates a static dust surface once at initialization and stores dust positions.

        Returns:
            pygame.Surface: Surface with dust particles drawn on it
        """
        dust_particles = []  # Store dust particle positions

        # Set dust density (percentage of empty cells that will have dust)
        dust_density = 0.1
        dust_size_range = (2, 2)

        # Iterate through grid cells
        for i in range(height):
            for j in range(width):
                # Only add dust to empty cells
                if grid[i, j] == 0:
                    # Calculate how many dust particles to add to this cell
                    dust_count = int(dust_density * cell_size)

                    # Calculate cell position
                    cell_left = j * cell_size
                    cell_top = i * cell_size

                    for _ in range(dust_count):
                        # Random position within the cell
                        dust_x = cell_left + random.randint(0, cell_size)
                        dust_y = cell_top + random.randint(0, cell_size)

                        # Random dust properties
                        dust_size = random.randint(dust_size_range[0], dust_size_range[1])
                        dust_alpha = random.randint(20, 100)  # Random transparency

                        # Create dust color with random alpha
                        dust_color = (*color, dust_alpha)

                        # Store the dust particle data
                        dust_particles.append({
                            'pos': (dust_x, dust_y),
                            'size': dust_size,
                            'color': dust_color,
                            'collected': False
                        })

        return dust_particles