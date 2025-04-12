import sys
import pygame
import numpy as np
import math
from config import Config


class Picasso:
    def __init__(self, screen: pygame.Surface):
        self.medium_font = pygame.font.SysFont(None, 24)  # for rendering text
        self.small_font = pygame.font.SysFont(None, 20)
        # Define the Search and Rescue theme's starting point and safe zone
        self.starting_point = np.array([Config.CELL_SIZE * 1.5, Config.CELL_SIZE * 1.5])
        self.safe_zone = np.array([540, 460])  # designated safe zone (end goal)
        self.screen = screen
        self.clock = pygame.time.Clock()


    def draw_map(self, robot, show_sensors=False):
        self._draw_maze()
        if show_sensors: self._draw_sensor_readings(robot)
        self._draw_visible_landmarks(robot)
        self._draw_robot(robot)
        self._draw_velocities(robot.left_velocity, robot.right_velocity, robot.theta)
        self._draw_path_history(robot.path_history)
        self._draw_landmarks()
        # Highlight collision if one occurred
        if robot.last_collision_cell:
            self._draw_collision_marker(robot.last_collision_cell, robot.x, robot.y)

    def _draw_maze(self):
        self.screen.fill(Config.WHITE)
        # Draw the maze
        for i in range(Config.GRID_HEIGHT):
            for j in range(Config.GRID_WIDTH):
                rect = pygame.Rect(j * Config.CELL_SIZE, i * Config.CELL_SIZE, Config.CELL_SIZE, Config.CELL_SIZE)
                if Config.maze_grid[i, j] == 1:
                    pygame.draw.rect(self.screen, Config.BLACK, rect)
                else:
                    pygame.draw.rect(self.screen, Config.GRAY, rect)

    def _draw_robot(self, robot):
        x = int(robot.x)
        y = int(robot.y)
        pygame.draw.circle(self.screen, Config.BLUE, (x, y), robot.radius)
        end_x = x + int(robot.radius * math.cos(robot.theta))
        end_y = y + int(robot.radius * math.sin(robot.theta))
        pygame.draw.line(self.screen, Config.RED, (x, y), (end_x, end_y), 2)

    def _draw_sensor_readings(self, robot):
        sensor_readings = robot.get_sensor_readings(Config.maze_grid)
        for i, reading in enumerate(sensor_readings):
            text = self.small_font.render(f"{reading:.0f}", True, Config.RED)
            angle = robot.theta + robot.sensor_angles[i]
            text_x = int(robot.x + (reading) * math.cos(angle))
            text_y = int(robot.y + (reading) * math.sin(angle))
            self.screen.blit(text, (text_x, text_y))
            pygame.draw.line(self.screen, Config.GREEN, (int(robot.x), int(robot.y)), (text_x, text_y), 1)

    def _draw_path_history(self, path_history, color=(0, 255, 255)):
        if len(path_history) < 2:
            return

            # Create a transparent surface for alpha blending
        trail_surface = pygame.Surface((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT), pygame.SRCALPHA)

        trail_points = [(int(x), int(y)) for x, y in path_history]
        history_len = len(trail_points)

        for i in range(history_len - 1):
            # Fade older points more
            alpha = int(255 * (i / history_len))
            faded_color = (*Config.AQUA, alpha)  # Add alpha to RGB color
            pygame.draw.line(trail_surface, faded_color, trail_points[i], trail_points[i + 1], 2)

        # Blit the trail surface onto the main screen
        self.screen.blit(trail_surface, (0, 0))

    def _draw_velocities(self, l_v, r_v, theta):
        vel_text = self.small_font.render(f"l_vel: x={l_v:.1f} | r_vel={r_v:.1f} | θ={theta:.1f}",True, Config.RED)
        self.screen.blit(vel_text, (Config.WINDOW_WIDTH - 220, 20))

    def _draw_landmarks(self):
        tile_size = Config.CELL_SIZE
        for landmark in Config.landmarks:
            cx, cy = landmark
            pygame.draw.circle(self.screen, Config.ORANGE, (cx, cy), tile_size // 10)


    def _draw_collision_marker(self, cell_pos: tuple[int, int], robot_x: float, robot_y: float):
        """
        Draw a neon-colored line on the side of the obstacle cell that the robot is colliding with.
        """
        i, j = cell_pos
        cell_size = Config.CELL_SIZE
        cell_left = j * cell_size
        cell_top = i * cell_size
        cell_center_x = cell_left + cell_size / 2
        cell_center_y = cell_top + cell_size / 2

        # Determine the direction from robot to cell center
        dx = robot_x - cell_center_x
        dy = robot_y - cell_center_y

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        thickness = 4
        pad = 6

        if abs_dx > abs_dy:
            # Horizontal collision
            if dx > 0:
                # Collision on right side
                start = (cell_left + cell_size - pad, cell_top + pad)
                end = (cell_left + cell_size - pad, cell_top + cell_size - pad)
            else:
                # Collision on left side
                start = (cell_left + pad, cell_top + pad)
                end = (cell_left + pad, cell_top + cell_size - pad)
        else:
            # Vertical collision
            if dy > 0:
                # Collision on bottom
                start = (cell_left + pad, cell_top + cell_size - pad)
                end = (cell_left + cell_size - pad, cell_top + cell_size - pad)
            else:
                # Collision on top
                start = (cell_left + pad, cell_top + pad)
                end = (cell_left + cell_size - pad, cell_top + pad)

        pygame.draw.line(self.screen, Config.NEON_PINK, start, end, thickness)

    def update_display(self, fps):
        pygame.display.flip()
        self.clock.tick(fps)

    @staticmethod
    def quit():
        pygame.quit()
        sys.exit()

    def _draw_visible_landmarks(self, robot):
        """
        Draw lines from the robot to landmarks that are within sensor range and visible
        (not occluded by obstacles).
        """
        # Extract robot position and sensor range
        robot_x, robot_y = robot.x, robot.y
        sensor_range = robot.sensor_range

        for landmark in Config.landmarks:
            lm_x, lm_y = landmark

            # Calculate Euclidean distance to the landmark
            distance = math.sqrt((lm_x - robot_x) ** 2 + (lm_y - robot_y) ** 2)

            # Check if landmark is within sensor range
            if distance <= sensor_range:
                # Check if there's a clear line of sight (no obstacles)
                is_visible = True
                cell_size = Config.CELL_SIZE

                # Use Bresenham's line algorithm to check visibility
                dx = abs(lm_x - robot_x)
                dy = abs(lm_y - robot_y)
                sx = 1 if robot_x < lm_x else -1
                sy = 1 if robot_y < lm_y else -1
                err = dx - dy

                x, y = robot_x, robot_y

                while not (abs(x - lm_x) < 1 and abs(y - lm_y) < 1):
                    # Convert to grid coordinates
                    cell_x = int(x // cell_size)
                    cell_y = int(y // cell_size)

                    # Check if current cell is valid and not an obstacle
                    if (0 <= cell_x < Config.GRID_WIDTH and
                            0 <= cell_y < Config.GRID_HEIGHT and
                            Config.maze_grid[cell_y, cell_x] == 1):
                        is_visible = False
                        break

                    # Calculate next point
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x += sx
                    if e2 < dx:
                        err += dx
                        y += sy

                # If landmark is visible, draw a line to it
                if is_visible:
                    pygame.draw.line(self.screen, Config.GREEN,
                                     (int(robot_x), int(robot_y)),
                                     (int(lm_x), int(lm_y)), 1)
                    # Draw a small circle around visible landmarks
                    pygame.draw.circle(self.screen, Config.RED,
                                       (int(lm_x), int(lm_y)),
                                       Config.CELL_SIZE // 8, 2)
    

    def get_visible_landmark_measurements(self, robot):
        """
        Returns a list of (z, landmark_pos) for visible landmarks.
        z is the actual measurement: [distance, bearing]
        """
        visible_measurements = []
        robot_x, robot_y = robot.x, robot.y
        sensor_range = robot.sensor_range
        theta = robot.theta
        cell_size = Config.CELL_SIZE

        for lm_x, lm_y in Config.landmarks:
            dx = lm_x - robot_x
            dy = lm_y - robot_y
            distance = math.sqrt(dx**2 + dy**2)

            if distance <= sensor_range:
                # Bresenham's line-of-sight check (copied from _draw_visible_landmarks)
                is_visible = True
                x, y = robot_x, robot_y
                err = abs(dx) - abs(dy)
                sx = 1 if robot_x < lm_x else -1
                sy = 1 if robot_y < lm_y else -1
                while not (abs(x - lm_x) < 1 and abs(y - lm_y) < 1):
                    cell_x = int(x // cell_size)
                    cell_y = int(y // cell_size)
                    if (0 <= cell_x < Config.GRID_WIDTH and
                            0 <= cell_y < Config.GRID_HEIGHT and
                            Config.maze_grid[cell_y, cell_x] == 1):
                        is_visible = False
                        break
                    e2 = 2 * err
                    if e2 > -abs(dy):
                        err -= abs(dy)
                        x += sx
                    if e2 < abs(dx):
                        err += abs(dx)
                        y += sy

                if is_visible:
                    bearing = math.atan2(dy, dx) - theta
                    z = np.array([distance, ((bearing + math.pi) % (2 * math.pi)) - math.pi])  # normalized
                    visible_measurements.append((z, np.array([lm_x, lm_y])))

        return visible_measurements
    


    