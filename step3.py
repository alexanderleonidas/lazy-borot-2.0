import pygame
import sys
import math
from config import Config
from robot import Robot, Action
from filter import EKF
from mapping import OccupancyGrid

#############################################
# Main Simulation Loop (Steps 1, 2, and 3)   #
#############################################

def main():
    pygame.init()
    screen = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
    pygame.display.set_caption("Robot Simulator with EKF Localization and Mapping (Step 3)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    # Create a robot instance at a starting position.
    robot = Robot(Config.CELL_SIZE * 1.5, Config.CELL_SIZE * 1.5, 0)
    # Initialize the EKF with the same initial state.
    ekf = EKF(robot.get_pose().copy())
    # Initialize the occupancy grid (mapping)
    occ_grid = OccupancyGrid(Config.GRID_WIDTH, Config.GRID_HEIGHT)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Keyboard Controls for robot movement ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            robot.set_velocity(Action.INCREASE_LEFT)
        elif keys[pygame.K_s]:
            robot.set_velocity(Action.DECREASE_LEFT)
        elif keys[pygame.K_o]:
            robot.set_velocity(Action.INCREASE_RIGHT)
        elif keys[pygame.K_k]:
            robot.set_velocity(Action.DECREASE_RIGHT)
        elif keys[pygame.K_SPACE]:
            robot.set_velocity(Action.BREAK)
        else:
            robot.set_velocity(Action.NOTHING)

        dt = 0.1
        control = [robot.get_linear_velocity(), robot.get_angular_velocity()]
        ekf.predict(control, dt)
        robot.update(dt, Config.maze_grid)

        # Obtain sensor readings; use front sensor (index 1) for EKF update.
        sensor_readings = robot.get_sensor_readings(Config.maze_grid)
        measurement = sensor_readings[1]
        ekf.update(measurement, Config.maze_grid, sensor_angle=0)

        # Update the occupancy grid mapping.
        occ_grid.update(robot, sensor_readings)

        ###################
        # Rendering Phase #
        ###################
        screen.fill(Config.WHITE)
        # Draw occupancy grid mapping (overlay)
        for i in range(occ_grid.height):
            for j in range(occ_grid.width):
                occ_val = occ_grid.grid[i, j]
                # Define color based on occupancy probability:
                if occ_val == 0.0:
                    color = (220, 220, 220)  # free space: light
                elif occ_val == 1.0:
                    color = (50, 50, 50)     # occupied: dark
                else:
                    color = (150, 150, 150)  # unknown: medium gray
                rect = pygame.Rect(j * Config.CELL_SIZE, i * Config.CELL_SIZE, Config.CELL_SIZE, Config.CELL_SIZE)
                pygame.draw.rect(screen, color, rect)

        # Draw the maze boundaries on top of the occupancy grid
        for i in range(Config.GRID_HEIGHT):
            for j in range(Config.GRID_WIDTH):
                rect = pygame.Rect(j * Config.CELL_SIZE, i * Config.CELL_SIZE, Config.CELL_SIZE, Config.CELL_SIZE)
                if Config.maze_grid[i, j] == 1:
                    pygame.draw.rect(screen, Config.BLACK, rect)

        # Draw the true robot pose (blue circle with red heading)
        pygame.draw.circle(screen, Config.BLUE, (int(robot.x), int(robot.y)), robot.radius)
        true_end_x = int(robot.x + robot.radius * math.cos(robot.theta))
        true_end_y = int(robot.y + robot.radius * math.sin(robot.theta))
        pygame.draw.line(screen, Config.RED, (int(robot.x), int(robot.y)), (true_end_x, true_end_y), 2)

        # Draw the EKF estimated pose (yellow circle)
        est = ekf.state
        pygame.draw.circle(screen, Config.YELLOW, (int(est[0]), int(est[1])), robot.radius // 2)
        est_end_x = int(est[0] + (robot.radius // 2) * math.cos(est[2]))
        est_end_y = int(est[1] + (robot.radius // 2) * math.sin(est[2]))
        pygame.draw.line(screen, Config.RED, (int(est[0]), int(est[1])), (est_end_x, est_end_y), 2)

        # Optionally, display sensor readings near the robot
        for i, reading in enumerate(sensor_readings):
            angle = robot.theta + robot.sensor_angles[i]
            text = font.render(f"{reading:.0f}", True, Config.RED)
            text_x = int(robot.x + (reading + 10) * math.cos(angle))
            text_y = int(robot.y + (reading + 10) * math.sin(angle))
            screen.blit(text, (text_x, text_y))

        # Display EKF estimated state
        est_text = font.render(f"EKF: x={est[0]:.1f}, y={est[1]:.1f}, θ={math.degrees(est[2]):.1f}°", True, Config.RED)
        screen.blit(est_text, (10, Config.WINDOW_HEIGHT - 30))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()