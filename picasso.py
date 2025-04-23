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

    def draw_map(self, robot, show_sensors=False, belief_history=None):
        self._draw_maze()
        if show_sensors: self._draw_sensor_readings(robot)
        self._draw_visible_landmarks(robot)

        # Draw ground truth robot position last (or optionally disable)
        self._draw_robot(robot)

        self._draw_velocities(robot.left_velocity, robot.right_velocity, robot.theta)
        self._draw_path_history(robot.path_history)  # Ground truth path
        self._draw_landmarks()

        # Draw Kalman Filter related elements
        if hasattr(robot, 'filter') and robot.filter:
            # Draw uncertainty ellipse based on filter's covariance
            self._draw_uncertainty_ellipse(robot)
            # Draw the estimated pose from the filter
            # self._draw_estimated_pose(robot.filter.pose)
            # Draw belief history if available
            if belief_history:
                self._draw_belief_history(belief_history)

        # Highlight the collision if one occurred
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
        sys.exit()

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

        # Convert each belief to a 2D point ignoring the orientation.
        points = [(int(pose[0]), int(pose[1])) for pose in belief_history]

        draw_segment = True
        segment_count = 0

        for i in range(len(points) - 1):
            if draw_segment:
                pygame.draw.line(self.screen, Config.PURPLE, points[i], points[i + 1], 2) # Cyan color

            segment_count += 1

            if draw_segment and segment_count >= dash_length:
                draw_segment = False
                segment_count = 0
            elif not draw_segment and segment_count >= gap_length:
                draw_segment = True
                segment_count = 0

    def _draw_uncertainty_ellipse(self, robot, confidence_sigma=2.0):
        """
        Draws the positional uncertainty ellipse based on the Kalman filter's covariance.
        Also stores the ellipse parameters for later reference.
        """
        # Get ellipse parameters from the filter
        ellipse_params = robot.filter.calculate_uncertainty_ellipse(confidence_sigma)

        if ellipse_params is None:
            # print("\rCannot draw ellipse: Invalid parameters.", end='', flush=True)
            return  # Cannot draw if parameters are invalid

        semi_major, semi_minor, angle_deg = ellipse_params

        # Store ellipse parameters in the robot's filter for potential use elsewhere
        if not hasattr(robot.filter, 'uncertainty_history'):
            robot.filter.uncertainty_history = []

        # Ensure axes are minimally visible if very small
        width = max(int(2 * semi_major), 2)  # Minimum width of 2 pixels
        height = max(int(2 * semi_minor), 2)  # Minimum height of 2 pixels

        # Center of ellipse is the filter's current estimated position
        center_x, center_y = int(robot.filter.pose[0]), int(robot.filter.pose[1])

        # --- Draw Rotated Ellipse using a temporary surface ---
        # 1. Create a surface large enough for the ellipse
        #    Make it slightly larger than max(width, height) to handle rotation without clipping
        surface_size = int(max(width, height) * 1.5)
        # Ensure surface size is at least 1x1
        if surface_size <= 0: surface_size = max(width, height, 2)

        try:
            ellipse_surface = pygame.Surface((surface_size, surface_size),
                                             pygame.SRCALPHA)  # Use SRCALPHA for transparency
        except pygame.error as e:
            print(f"\rPygame error creating surface ({surface_size}x{surface_size}): {e}", end='', flush=True)
            return  # Cannot create surface, likely too large or zero sizes

        # 2. Define the ellipse's rect on this temporary surface, centered
        ellipse_rect = pygame.Rect(0, 0, width, height)
        ellipse_rect.center = (surface_size // 2, surface_size // 2)

        # 3. Draw the ellipse onto the temporary surface
        ellipse_color = (*Config.ORANGE, 60)  # Use Orange with low alpha for fill
        ellipse_outline_color = (*Config.ORANGE, 120)  # Darker/less transparent outline

        try:
            pygame.draw.ellipse(ellipse_surface, ellipse_color, ellipse_rect)  # Filled ellipse
            pygame.draw.ellipse(ellipse_surface, ellipse_outline_color, ellipse_rect, 1)  # Outline (width 1)
        except pygame.error as e:
            print(f"\rPygame error drawing ellipse (w={width}, h={height}): {e}", end='', flush=True)
            return  # Error during drawing

        # 4. Rotate the temporary surface containing the ellipse
        #    Pygame rotates counter-clockwise, so use the negative angle if angle_deg is clockwise
        #    Our angle_deg is from positive x-axis CCW, matching Pygame.
        rotated_surface = pygame.transform.rotate(ellipse_surface, angle_deg)
        rotated_rect = rotated_surface.get_rect(center=(center_x, center_y))

        # 5. Blit the rotated surface onto the main screen
        self.screen.blit(rotated_surface, rotated_rect)
