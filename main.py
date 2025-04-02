import sys
import pygame
import math
import numpy as np
from config import Config
from robot import Robot
from filter import EKF
from mapping import OccupancyGrid
from particle_swarm_optimization import PSOPlanner

#############################################
# Main Simulation Loop and Integration      #
#############################################

def main():
    pygame.init()
    font = pygame.font.SysFont(None, 24)  # for rendering text
    # Define the Search and Rescue theme's starting point and safe zone
    starting_point = np.array([Config.CELL_SIZE * 1.5, Config.CELL_SIZE * 1.5])
    safe_zone = np.array([540, 460])  # designated safe zone (end goal)
    screen = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
    pygame.display.set_caption("Autonomous Maze-Navigating Robot using PSO")
    clock = pygame.time.Clock()

    # Initialize the robot at the starting point
    start_x = starting_point[0]
    start_y = starting_point[1]
    robot = Robot(start_x, start_y, 0)

    # Initialize EKF with the same starting state
    ekf = EKF(np.array([start_x, start_y, 0]))

    # Create occupancy grid map
    occ_grid = OccupancyGrid(Config.GRID_WIDTH, Config.GRID_HEIGHT)

    # Instantiate PSO planner
    pso_planner = PSOPlanner(occ_grid)

    dt = 0.1  # time step (seconds)

    running = True
    while running:
        clock.tick(30)  # simulation at 30 FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Obtain sensor readings using ray-casting (with noise)
        sensor_readings = robot.get_sensor_readings(Config.maze_grid)
        # Update occupancy grid mapping with new sensor data
        occ_grid.update(robot, sensor_readings)

        # PSO planning: compute a new target each iteration based on current robot position
        robot_pos = np.array([robot.x, robot.y])
        target = pso_planner.plan(robot_pos)

        # Simple proportional controller to drive toward the target
        error = target - robot_pos
        distance_error = np.linalg.norm(error)
        # Debug output to trace values
        print(f"Robot pos: {robot_pos}, Target: {target}, Distance error: {distance_error}")
        angle_to_target = math.atan2(error[1], error[0])
        angle_error = angle_to_target - robot.theta
        # Normalize angle_error to [-pi, pi]
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        linear_vel = 100 if distance_error > 10 else 0  # start moving if error > 10 pixels
        angular_vel = 2 * angle_error
        angular_vel = 2 * angle_error
        angular_vel = max(min(angular_vel, 1.5), -1.5)  # limit turning
        robot.set_velocity(linear_vel, angular_vel)

        # EKF Prediction and Update using control inputs and sensor readings
        ekf.predict([linear_vel, angular_vel], dt)
        ekf.update(sensor_readings, robot.sensor_angles, Config.maze_grid, robot_pos)

        # Update robot’s pose with collision handling
        robot.update(dt, Config.maze_grid)
        # Check if the robot has reached the safe zone (within 15 pixels)
        if np.linalg.norm(np.array([robot.x, robot.y]) - safe_zone) < 15:
            print("Safe zone reached. Mission complete!")
            running = False

        ###################
        # Rendering Phase #
        ###################
        screen.fill(Config.WHITE)
        # Draw maze cells
        for i in range(Config.GRID_HEIGHT):
            for j in range(Config.GRID_WIDTH):
                rect = pygame.Rect(j*Config.CELL_SIZE, i*Config.CELL_SIZE, Config.CELL_SIZE, Config.CELL_SIZE)
                if Config.maze_grid[i, j] == 1:
                    pygame.draw.rect(screen, Config.BLACK, rect)
                else:
                    # Overlay occupancy grid colors:
                    occ_val = occ_grid.grid[i, j]
                    if occ_val == 0.0:
                        color = (200, 200, 200)  # free
                    elif occ_val == 1.0:
                        color = (50, 50, 50)     # occupied
                    else:
                        color = (150, 150, 150)  # unknown
                    pygame.draw.rect(screen, color, rect)

        # Draw the starting point marker (green circle) and label
        pygame.draw.circle(screen, (0, 255, 0), (int(starting_point[0]), int(starting_point[1])), 8)
        start_label = font.render("Start", True, (0, 255, 0))
        screen.blit(start_label, (int(starting_point[0]) - 20, int(starting_point[1]) - 20))

        # Draw the safe zone marker (magenta circle) and label
        pygame.draw.circle(screen, (255, 0, 255), (int(safe_zone[0]), int(safe_zone[1])), 8)
        safe_label = font.render("Safe Zone", True, (255, 0, 255))
        screen.blit(safe_label, (int(safe_zone[0]) - 20, int(safe_zone[1]) - 20))

        # Draw the robot as a blue circle with a red line indicating heading.
        pygame.draw.circle(screen, Config.BLUE, (int(robot.x), int(robot.y)), robot.radius)
        end_x = int(robot.x + robot.radius * math.cos(robot.theta))
        end_y = int(robot.y + robot.radius * math.sin(robot.theta))
        pygame.draw.line(screen, Config.RED, (int(robot.x), int(robot.y)), (end_x, end_y), 2)

        # Draw sensor rays in green.
        for i, angle in enumerate(robot.sensor_angles):
            sensor_angle = robot.theta + angle
            ray_end_x = int(robot.x + sensor_readings[i] * math.cos(sensor_angle))
            ray_end_y = int(robot.y + sensor_readings[i] * math.sin(sensor_angle))
            pygame.draw.line(screen, Config.GREEN, (int(robot.x), int(robot.y)), (ray_end_x, ray_end_y), 1)

        # Draw the current target as a yellow circle.
        if target is not None:
            pygame.draw.circle(screen, Config.YELLOW, (int(target[0]), int(target[1])), 5)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()