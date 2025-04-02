import pygame
import sys
import numpy as np
import math
from picasso import Picasso
from config import Config
from robot import Robot
from filter import EKF
from mapping import OccupancyGrid
from particle_swarm_optimization import PSOPlanner

#############################################
# Main Simulation Loop (Steps 1, 2, 3, 4)   #
#############################################

def main():
    pygame.init()
    screen = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
    pygame.display.set_caption("Autonomous Navigation with Localization and Mapping (Step 4)")
    picasso = Picasso(screen)

    # Initialize robot, EKF, and occupancy grid.
    robot = Robot(Config.CELL_SIZE * 1.5, Config.CELL_SIZE * 1.5, 0)
    ekf = EKF(robot.get_pose().copy())
    occ_grid = OccupancyGrid(Config.GRID_WIDTH, Config.GRID_HEIGHT)
    # Instantiate PSO planner using the occupancy grid.
    pso_planner = PSOPlanner(occ_grid)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dt = 0.1
        # Autonomous Navigation using PSO:
        robot_pos = np.array([robot.x, robot.y])
        target = pso_planner.plan(robot_pos)
        # Compute error vector and angle to target.
        error = target - robot_pos
        distance_error = np.linalg.norm(error)
        angle_to_target = math.atan2(error[1], error[0])
        angle_error = angle_to_target - robot.theta
        # Normalize angle_error to [-pi, pi]
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        # Set control commands based on target error.
        linear_vel = 100 if distance_error > 10 else 0
        angular_vel = 2 * angle_error
        robot.set_velocity(linear_vel, angular_vel)

        # EKF prediction using the control commands.
        control = [robot.linear_velocity, robot.angular_velocity]
        ekf.predict(control, dt)
        robot.update(dt, Config.maze_grid)

        # Get sensor readings and update EKF using the front sensor (index 1).
        sensor_readings = robot.get_sensor_readings(Config.maze_grid)
        measurement = sensor_readings[1]
        ekf.update(measurement, Config.maze_grid, sensor_angle=0)

        # Update occupancy grid (mapping).
        occ_grid.update(robot, sensor_readings)

        ###################
        # Rendering Phase #
        ###################
        screen.fill(Config.WHITE)
        # Draw occupancy grid mapping (overlay)
        for i in range(occ_grid.height):
            for j in range(occ_grid.width):
                occ_val = occ_grid.grid[i, j]
                if occ_val == 0.0:
                    color = (220, 220, 220)  # free space: light
                elif occ_val == 1.0:
                    color = (50, 50, 50)     # occupied: dark
                else:
                    color = (150, 150, 150)  # unknown: medium gray
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)

        # Draw maze boundaries on top of the occupancy grid.
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if maze_grid[i, j] == 1:
                    pygame.draw.rect(screen, BLACK, rect)

        # Draw the true robot pose (blue circle with red heading).
        pygame.draw.circle(screen, BLUE, (int(robot.x), int(robot.y)), robot.radius)
        true_end_x = int(robot.x + robot.radius * math.cos(robot.theta))
        true_end_y = int(robot.y + robot.radius * math.sin(robot.theta))
        pygame.draw.line(screen, RED, (int(robot.x), int(robot.y)), (true_end_x, true_end_y), 2)

        # Draw the EKF estimated pose (yellow circle).
        est = ekf.state
        pygame.draw.circle(screen, YELLOW, (int(est[0]), int(est[1])), robot.radius // 2)
        est_end_x = int(est[0] + (robot.radius // 2) * math.cos(est[2]))
        est_end_y = int(est[1] + (robot.radius // 2) * math.sin(est[2]))
        pygame.draw.line(screen, RED, (int(est[0]), int(est[1])), (est_end_x, est_end_y), 2)

        # Draw the navigation target (cyan circle).
        pygame.draw.circle(screen, CYAN, (int(target[0]), int(target[1])), 5)

        # Optionally, display sensor readings near the robot.
        for i, reading in enumerate(sensor_readings):
            angle = robot.theta + robot.sensor_angles[i]
            text = font.render(f"{reading:.0f}", True, RED)
            text_x = int(robot.x + (reading + 10) * math.cos(angle))
            text_y = int(robot.y + (reading + 10) * math.sin(angle))
            screen.blit(text, (text_x, text_y))

        # Display EKF estimated state.
        est_text = font.render(f"EKF: x={est[0]:.1f}, y={est[1]:.1f}, θ={math.degrees(est[2]):.1f}°", True, RED)
        screen.blit(est_text, (10, WINDOW_HEIGHT - 30))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()