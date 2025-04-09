import numpy as np
import random

def generate_bordered_map(width: int, height: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    return grid

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