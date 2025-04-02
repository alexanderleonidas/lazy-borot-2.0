import pygame
import sys
import numpy as np
import math
import random

#####################################
# Global Settings and Maze Creation #
#####################################

# Maze grid: 0 = free, 1 = wall
maze_grid = np.array([
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1],
    [1,0,1,0,1,0,1,1,1,1,0,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,1],
    [1,0,1,1,1,1,1,0,0,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,1,0,0,0,0,1,0,1,0,1],
    [1,1,1,1,1,0,1,1,1,1,0,1,0,1,0,1],
    [1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1],
    [1,0,1,0,1,1,1,1,0,1,1,1,1,1,0,1],
    [1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1],
    [1,0,1,1,1,1,0,1,1,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
])
GRID_HEIGHT, GRID_WIDTH = maze_grid.shape
CELL_SIZE = 40  # pixels per cell

# Pygame window size
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

# Define colors
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GRAY   = (200, 200, 200)
BLUE   = (0, 0, 255)
RED    = (255, 0, 0)
GREEN  = (0, 255, 0)

#########################
# Milestone 1: Simulator #
#########################

class Robot:
    def __init__(self, x, y, theta):
        # Initial pose (x, y, theta)
        self.x = x  
        self.y = y
        self.theta = theta  # in radians

        self.radius = 10  # for drawing the robot
        self.linear_velocity = 0.0  # pixels per second
        self.angular_velocity = 0.0  # radians per second
        self.wheel_base = 20  # for differential drive simulation

        # Sensor configuration (simulating 3 sensors; can be extended to 12)
        self.sensor_range = 100  # max range in pixels
        self.sensor_angles = [-math.pi/4, 0, math.pi/4]  # relative sensor directions

    def update(self, dt, maze):
        """
        Update the robot's pose using axis-separated collision detection.
        This method computes the new x and y positions separately and checks
        for collisions along each axis to allow the robot to slide along walls.
        """
        # --- Update x-axis ---
        proposed_x = self.x + self.linear_velocity * math.cos(self.theta) * dt
        cell_x = int(proposed_x // CELL_SIZE)
        cell_y = int(self.y // CELL_SIZE)
        # If the proposed x position is in free space, update x.
        if 0 <= cell_x < GRID_WIDTH and 0 <= cell_y < GRID_HEIGHT and maze[cell_y, cell_x] == 0:
            self.x = proposed_x
        # --- Update y-axis ---
        proposed_y = self.y + self.linear_velocity * math.sin(self.theta) * dt
        cell_x = int(self.x // CELL_SIZE)
        cell_y = int(proposed_y // CELL_SIZE)
        # If the proposed y position is in free space, update y.
        if 0 <= cell_x < GRID_WIDTH and 0 <= cell_y < GRID_HEIGHT and maze[cell_y, cell_x] == 0:
            self.y = proposed_y

        # Always update the orientation
        self.theta += self.angular_velocity * dt
        self.theta %= 2 * math.pi

    def set_velocity(self, linear, angular):
        """
        Set the robot's linear and angular velocities.
        """
        self.linear_velocity = linear
        self.angular_velocity = angular

    def get_pose(self):
        """
        Return the current pose as a numpy array: [x, y, theta].
        """
        return np.array([self.x, self.y, self.theta])

    def get_sensor_readings(self, maze):
        """
        Simulate sensor readings using ray-casting.
        A small Gaussian noise is added to each measurement.
        Returns a list of distances (one per sensor).
        """
        readings = []
        noise_std = 1.0  # standard deviation for sensor noise
        for angle in self.sensor_angles:
            sensor_angle = self.theta + angle
            distance = 0
            while distance < self.sensor_range:
                test_x = int((self.x + distance * math.cos(sensor_angle)) // CELL_SIZE)
                test_y = int((self.y + distance * math.sin(sensor_angle)) // CELL_SIZE)
                if test_x < 0 or test_x >= GRID_WIDTH or test_y < 0 or test_y >= GRID_HEIGHT:
                    break
                if maze[test_y, test_x] == 1:  # wall detected
                    break
                distance += 1
            # Add noise to the sensor reading
            noisy_distance = max(0, distance + np.random.normal(0, noise_std))
            readings.append(noisy_distance)
        return readings

#############################################
# Main Simulation Loop for Step 1 Simulator  #
#############################################

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Robot Simulator - Step 1")
    clock = pygame.time.Clock()

    # Create a robot instance at the starting position.
    robot = Robot(CELL_SIZE * 1.5, CELL_SIZE * 1.5, 0)

    # Main simulation loop
    running = True
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Keyboard Controls ---
        # Up/Down arrows control forward/backward movement.
        # Left/Right arrows control rotation.
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            robot.set_velocity(100, robot.angular_velocity)
        elif keys[pygame.K_DOWN]:
            robot.set_velocity(-100, robot.angular_velocity)
        else:
            # Stop linear movement if no up/down key is pressed.
            robot.set_velocity(0, robot.angular_velocity)

        if keys[pygame.K_LEFT]:
            robot.angular_velocity = -2  # rotate left
        elif keys[pygame.K_RIGHT]:
            robot.angular_velocity = 2   # rotate right
        else:
            robot.angular_velocity = 0   # no rotation

        # Update the robot's state with a fixed time step.
        dt = 0.1
        robot.update(dt, maze_grid)

        # --- Rendering ---
        screen.fill(WHITE)
        # Draw the maze
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if maze_grid[i, j] == 1:
                    pygame.draw.rect(screen, BLACK, rect)
                else:
                    pygame.draw.rect(screen, GRAY, rect)

        # Draw the robot as a blue circle
        pygame.draw.circle(screen, BLUE, (int(robot.x), int(robot.y)), robot.radius)
        # Draw a red line to indicate the robot's heading
        end_x = int(robot.x + robot.radius * math.cos(robot.theta))
        end_y = int(robot.y + robot.radius * math.sin(robot.theta))
        pygame.draw.line(screen, RED, (int(robot.x), int(robot.y)), (end_x, end_y), 2)

        # Optionally, display sensor readings near the robot.
        sensor_readings = robot.get_sensor_readings(maze_grid)
        font = pygame.font.SysFont(None, 20)
        for i, reading in enumerate(sensor_readings):
            text = font.render(f"{reading:.0f}", True, RED)
            angle = robot.theta + robot.sensor_angles[i]
            text_x = int(robot.x + (reading + 10) * math.cos(angle))
            text_y = int(robot.y + (reading + 10) * math.sin(angle))
            screen.blit(text, (text_x, text_y))

        pygame.display.flip()
        clock.tick(30)  # run at 30 FPS

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()