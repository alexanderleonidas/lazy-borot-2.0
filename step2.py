import pygame
import math
from robot import Robot, Action
from config import Config
from filter import EKF
from picasso import Picasso


def main():
    pygame.init()
    screen = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
    pygame.display.set_caption("Robot Simulator with EKF Localization (Step 2)")
    picasso = Picasso(screen)

    # Create a robot instance at a starting position.
    robot = Robot(Config.CELL_SIZE * 1.5, Config.CELL_SIZE * 1.5, 0)
    # Initialize the EKF with the same initial state.
    ekf = EKF(robot.get_pose().copy())

    # Main simulation loop
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
        # Get control inputs used for EKF prediction (same as robot's current commands)
        control = [robot.get_linear_velocity(), robot.get_angular_velocity()]
        ekf.predict(control, dt)
        robot.update(dt, Config.maze_grid)

        # Obtain sensor readings (simulate measurement); use front sensor (index 1)
        sensor_readings = robot.get_sensor_readings(Config.maze_grid)
        measurement = sensor_readings[1]  # front sensor reading
        # EKF update using the front sensor (relative angle = 0)
        ekf.update(measurement, Config.maze_grid, sensor_angle=0)

        # --- Rendering ---
        picasso.draw_map(robot)

        # Draw the EKF estimated pose (yellow circle)
        est = ekf.state
        pygame.draw.circle(screen, Config.YELLOW, (int(est[0]), int(est[1])), robot.radius // 2)
        est_end_x = int(est[0] + (robot.radius//2) * math.cos(est[2]))
        est_end_y = int(est[1] + (robot.radius//2) * math.sin(est[2]))
        pygame.draw.line(screen, Config.RED, (int(est[0]), int(est[1])), (est_end_x, est_end_y), 2)

        # Display EKF estimated state
        est_text = picasso.small_font.render(f"EKF: x={est[0]:.1f}, y={est[1]:.1f}, θ={math.degrees(est[2]):.1f}°", True, Config.RED)
        screen.blit(est_text, (10, Config.WINDOW_HEIGHT - 30))

        picasso.update_display()
    picasso.quit()

if __name__ == "__main__":
    main()