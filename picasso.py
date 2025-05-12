import random
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
        self.dust_surface = self._generate_static_dust()

    def draw_map(self, robot, show_sensors=False, show_dust=False):
        self._draw_bitmap()
        if show_dust: self._draw_dust(robot)
        if hasattr(robot, 'mapping') and robot.mapping:
            self._draw_occupancy_grid(robot.mapping)
        # Draw ground truth robot position last (or optionally disable)
        if show_sensors: self._draw_sensor_readings(robot)
        self._draw_visible_landmarks(robot)
        self._draw_robot(robot)
        self._draw_path_history(robot.path_history)  # Ground truth path
        self._draw_landmarks()

        # Draw Kalman Filter related elements
        if hasattr(robot, 'filter') and robot.filter:
            # Draw uncertainty ellipse based on filter's covariance
            self._draw_uncertainty_ellipse_history(robot)
            # Draw belief history if available
            self._draw_belief_history(robot.filter.belief_history)

        # Highlight the collision if one occurred
        if robot.last_collision_cell:
            self._draw_collision_marker(robot.last_collision_cell, robot.x, robot.y)
        self._draw_velocities(robot.left_velocity, robot.right_velocity, robot.theta)

    def _draw_bitmap(self):
        self.screen.fill(Config.WHITE)
        # Draw the maze
        for i in range(Config.GRID_HEIGHT):
            for j in range(Config.GRID_WIDTH):
                rect = pygame.Rect(j * Config.CELL_SIZE, i * Config.CELL_SIZE, Config.CELL_SIZE, Config.CELL_SIZE)
                if Config.maze_grid[i, j] == 1:
                    pygame.draw.rect(self.screen, Config.BLACK, rect)
                else:
                    pygame.draw.rect(self.screen, Config.WHITE, rect)

    def _draw_robot(self, robot):
        x = int(robot.x)
        y = int(robot.y)
        pygame.draw.circle(self.screen, Config.BLUE, (x, y), robot.radius)
        end_x = x + int(robot.radius * math.cos(robot.theta))
        end_y = y + int(robot.radius * math.sin(robot.theta))
        pygame.draw.line(self.screen, Config.RED, (x, y), (end_x, end_y), 2)

    def _draw_sensor_readings(self, robot):
        for i, reading in enumerate(robot.sensor_readings):
            text = self.small_font.render(f"{reading[0]:.0f}", True, Config.RED)
            angle = robot.theta + robot.sensor_angles[i]
            text_x = int(robot.x + (reading[0]) * math.cos(angle))
            text_y = int(robot.y + (reading[0]) * math.sin(angle))
            self.screen.blit(text, (text_x, text_y))
            pygame.draw.line(self.screen, Config.GREEN, (int(robot.x), int(robot.y)), (text_x, text_y), 1)

    def _draw_path_history(self, path_history):
        if len(path_history) < 2:
            return

        # Limit to the last 200 points
        limited_history = path_history[-200:] if len(path_history) > 200 else path_history

        # Create a transparent surface for alpha blending
        trail_surface = pygame.Surface((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT), pygame.SRCALPHA)

        trail_points = [(int(x), int(y)) for x, y in limited_history]
        history_len = len(trail_points)

        for i in range(history_len - 1):
            # Fade older points more
            alpha = int(255 * (i / history_len))
            faded_color = (*Config.AQUA, alpha)  # Add alpha to RGB color
            pygame.draw.line(trail_surface, faded_color, trail_points[i], trail_points[i + 1], 2)

        # Blit the trail surface onto the main screen
        self.screen.blit(trail_surface, (0, 0))

    def _draw_estimated_pose(self, pose):
        # Estimated Robot Position from Filter
        x = int(pose[0])
        y = int(pose[1])
        theta = pose[2]
        radius = 8  # Make slightly smaller than ground truth maybe

        # Draw circle for estimated position
        pygame.draw.circle(self.screen, Config.PURPLE, (x, y), radius, 2)  # Draw outline
        # Line indicating estimated orientation
        end_x = x + int(radius * math.cos(theta))
        end_y = y + int(radius * math.sin(theta))
        pygame.draw.line(self.screen, Config.PURPLE, (x, y), (end_x, end_y), 2)

    def _draw_velocities(self, l_v, r_v, theta):
        vel_text = self.small_font.render(f"l_vel: x={l_v:.1f} | r_vel={r_v:.1f} | θ={theta:.1f}", True, Config.RED)
        self.screen.blit(vel_text, (Config.WINDOW_WIDTH - 220, 20))

    def _draw_landmarks(self):
        tile_size = Config.CELL_SIZE
        for landmark in Config.landmarks:
            cx, cy = landmark
            pygame.draw.circle(self.screen, Config.ORANGE, (cx, cy), tile_size //3.5)

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

        # Determine the direction from the robot to the cell center
        dx = robot_x - cell_center_x
        dy = robot_y - cell_center_y

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        thickness = 4
        pad = 6

        if abs_dx > abs_dy:
            # Horizontal collision
            if dx > 0:
                # Collision on the right side
                start = (cell_left + cell_size - pad, cell_top + pad)
                end = (cell_left + cell_size - pad, cell_top + cell_size - pad)
            else:
                # Collision on the left side
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

    def _draw_visible_landmarks(self, robot):
        """
        Draw lines from the robot to landmarks that are within sensor range and visible
        (not occluded by obstacles).
        """
        robot.get_visible_landmark_readings()
        for _, (lm_x, lm_y) in robot.visible_measurements:
            pygame.draw.line(self.screen, Config.GREEN,
                             (int(robot.x), int(robot.y)),
                             (int(lm_x), int(lm_y)), 1)
            pygame.draw.circle(self.screen, Config.RED,
                               (int(lm_x), int(lm_y)),
                               Config.CELL_SIZE // 8, 2)

    def _draw_belief_history(self, belief_history, dash_length=4, gap_length=3):
        """
        Draws the estimated trajectory (belief history) as a dashed Cyan line.

        Args:
            belief_history: List of estimated poses [x, y, theta].
            dash_length: Number of segments to draw for a dash.
            gap_length: Number of segments to skip for a gap.
        """
        if len(belief_history) < 2:
            return

        # Limit to the last 200 points
        limited_history = belief_history[-200:] if len(belief_history) > 200 else belief_history

        points = [(int(pose[0]), int(pose[1])) for pose in limited_history]

        draw_segment = True
        segment_count = 0

        for i in range(len(points) - 1):
            if draw_segment:
                pygame.draw.line(self.screen, Config.PURPLE, points[i], points[i + 1], 2)  # Cyan color

            segment_count += 1

            if draw_segment and segment_count >= dash_length:
                draw_segment = False
                segment_count = 0
            elif not draw_segment and segment_count >= gap_length:
                draw_segment = True
                segment_count = 0

    def _draw_uncertainty_ellipse_history(self, robot, max_history_draw=20, max_alpha_outline=100):
        """
        Draws a history of the robot's positional uncertainty ellipses (OUTLINE ONLY).
        Outlines fade out based on their age.

        Args:
            robot: The robot object containing the filter with uncertainty_history.
            max_history_draw (int): Maximum number of historical ellipses to draw.
            max_alpha_outline (int): Maximum alpha value (0-255) for the newest ellipse OUTLINE.
        """
        if not hasattr(robot.filter, 'uncertainty_history') or not robot.filter.uncertainty_history:
            return 

        history = robot.filter.uncertainty_history
        history_len = len(history)
        start_index = max(0, history_len - max_history_draw)
        drawable_history = history[start_index:]
        drawable_len = len(drawable_history)

        if drawable_len == 0:
            return

        for i, ellipse_data in enumerate(drawable_history):
            try:
                center_x, center_y = int(ellipse_data['center'][0]), int(ellipse_data['center'][1])
                semi_major = ellipse_data['semi_major']
                semi_minor = ellipse_data['semi_minor']
                angle_deg = ellipse_data['angle_deg']
            except (KeyError, IndexError, TypeError) as e:
                continue  

            age_ratio = (i + 1) / drawable_len
            current_alpha_outline = int(max_alpha_outline * age_ratio)
            current_alpha_outline = max(0, min(255, current_alpha_outline))

            if current_alpha_outline <= 1:
                continue

            width = max(int(2 * semi_major), 1)
            height = max(int(2 * semi_minor), 1)
            surface_size = int(max(width, height) * 1.5) + 2
            if surface_size <= 0: surface_size = max(width, height, 2)

            try:
                ellipse_surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
            except (pygame.error, ValueError) as e:
                continue 

            ellipse_rect = pygame.Rect(0, 0, width, height)
            ellipse_rect.center = (surface_size // 2, surface_size // 2)

            ellipse_outline_color = (*Config.ORANGE, current_alpha_outline)

            try:
                pygame.draw.ellipse(ellipse_surface, ellipse_outline_color, ellipse_rect, 1)
            except pygame.error as e:
                continue  

            rotated_surface = pygame.transform.rotate(ellipse_surface, angle_deg)
            rotated_rect = rotated_surface.get_rect(center=(center_x, center_y))
            self.screen.blit(rotated_surface, rotated_rect)

    def _draw_occupancy_grid(self, occupancy_grid):
        grayscale = occupancy_grid.get_grayscale_grid()
        surface = pygame.surfarray.make_surface(np.stack([grayscale] * 3, axis=-1).swapaxes(0, 1))
        surface = pygame.transform.scale(surface, (occupancy_grid.width * Config.CELL_SIZE, occupancy_grid.height * Config.CELL_SIZE))
        self.screen.blit(surface, (0, 0))

    def _generate_static_dust(self):
        """
        Generates a static dust surface once at initialization and stores dust positions.

        Returns:
            pygame.Surface: Surface with dust particles drawn on it
        """
        dust_surface = pygame.Surface((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT), pygame.SRCALPHA)
        self.dust_particles = []  # Store dust particle positions

        # Set dust density (percentage of empty cells that will have dust)
        dust_density = 0.1
        dust_size_range = (2, 2)

        # Iterate through grid cells
        for i in range(Config.GRID_HEIGHT):
            for j in range(Config.GRID_WIDTH):
                # Only add dust to empty cells
                if Config.maze_grid[i, j] == 0:
                    # Calculate how many dust particles to add to this cell
                    dust_count = int(dust_density * Config.CELL_SIZE)

                    # Calculate cell position
                    cell_left = j * Config.CELL_SIZE
                    cell_top = i * Config.CELL_SIZE

                    for _ in range(dust_count):
                        # Random position within the cell
                        dust_x = cell_left + random.randint(0, Config.CELL_SIZE)
                        dust_y = cell_top + random.randint(0, Config.CELL_SIZE)

                        # Random dust properties
                        dust_size = random.randint(dust_size_range[0], dust_size_range[1])
                        dust_alpha = random.randint(20, 100)  # Random transparency

                        # Create dust color with random alpha
                        dust_color = (*Config.GRAY, dust_alpha)

                        # Store the dust particle data
                        self.dust_particles.append({
                            'pos': (dust_x, dust_y),
                            'size': dust_size,
                            'color': dust_color,
                            'collected': False
                        })

                        # Draw the dust particle
                        pygame.draw.circle(dust_surface, dust_color, (dust_x, dust_y), dust_size)

        return dust_surface

    def _draw_dust(self, robot):
        """
        Draws dust particles and checks if the robot has collected any.

        Args:
            robot: The robot instance that can collect dust
        """
        # If robot doesn't have a dust_count attribute, add it
        if not hasattr(robot, 'dust_count'):
            robot.dust_count = 0

        # Create a new surface for dust with alpha blending
        updated_dust_surface = pygame.Surface((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT), pygame.SRCALPHA)

        # Check for collisions and redraw remaining dust
        for dust in self.dust_particles:
            if not dust['collected']:
                # Check if robot is over the dust
                dust_x, dust_y = dust['pos']
                dist = math.sqrt((robot.x - dust_x) ** 2 + (robot.y - dust_y) ** 2)

                if dist <= robot.radius:
                    # Robot has collected this dust particle
                    dust['collected'] = True
                    robot.dust_count += 1
                else:
                    # Dust hasn't been collected, so draw it
                    pygame.draw.circle(updated_dust_surface, dust['color'], dust['pos'], dust['size'])

        # Update the dust surface
        self.dust_surface = updated_dust_surface

        # Draw the dust surface on the screen
        self.screen.blit(self.dust_surface, (0, 0))

        # Display dust count
        dust_text = self.small_font.render(f"Dust Collected: {robot.dust_count}", True, Config.RED)
        self.screen.blit(dust_text, (10, 10))
