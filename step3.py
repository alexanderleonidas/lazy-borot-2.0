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
BLUE   = (0, 0, 255)      # True robot pose
RED    = (255, 0, 0)
GREEN  = (0, 255, 0)
YELLOW = (255, 255, 0)    # EKF estimated pose

#########################
# Step 1: Simulator     #
#########################

class Robot:
    def __init__(self, x, y, theta):
        # True robot state (ground truth)
        self.x = x  
        self.y = y
        self.theta = theta  # in radians

        self.radius = 10  # for drawing the robot
        self.linear_velocity = 0.0  # pixels per second
        self.angular_velocity = 0.0  # radians per second

        # Sensor configuration: simulating 3 sensors (front, left, right)
        self.sensor_range = 100  # max range in pixels
        self.sensor_angles = [-math.pi/4, 0, math.pi/4]  # relative sensor directions

    def update(self, dt, maze):
        """
        Update the robot's true pose using axis-separated collision detection.
        This allows the robot to slide along walls.
        """
        # Update x-axis component
        proposed_x = self.x + self.linear_velocity * math.cos(self.theta) * dt
        cell_x = int(proposed_x // CELL_SIZE)
        cell_y = int(self.y // CELL_SIZE)
        if 0 <= cell_x < GRID_WIDTH and 0 <= cell_y < GRID_HEIGHT and maze[cell_y, cell_x] == 0:
            self.x = proposed_x

        # Update y-axis component
        proposed_y = self.y + self.linear_velocity * math.sin(self.theta) * dt
        cell_x = int(self.x // CELL_SIZE)
        cell_y = int(proposed_y // CELL_SIZE)
        if 0 <= cell_x < GRID_WIDTH and 0 <= cell_y < GRID_HEIGHT and maze[cell_y, cell_x] == 0:
            self.y = proposed_y

        # Update orientation
        self.theta += self.angular_velocity * dt
        self.theta %= 2 * math.pi

    def set_velocity(self, linear, angular):
        """
        Set robot's linear and angular velocities.
        """
        self.linear_velocity = linear
        self.angular_velocity = angular

    def get_pose(self):
        """
        Return the true pose as a numpy array: [x, y, theta].
        """
        return np.array([self.x, self.y, self.theta])

    def get_sensor_readings(self, maze):
        """
        Simulate sensor readings using ray-casting.
        Gaussian noise is added to each measurement.
        Returns a list of distances (one per sensor).
        """
        readings = []
        noise_std = 1.0
        for angle in self.sensor_angles:
            sensor_angle = self.theta + angle
            distance = 0
            while distance < self.sensor_range:
                test_x = int((self.x + distance * math.cos(sensor_angle)) // CELL_SIZE)
                test_y = int((self.y + distance * math.sin(sensor_angle)) // CELL_SIZE)
                if test_x < 0 or test_x >= GRID_WIDTH or test_y < 0 or test_y >= GRID_HEIGHT:
                    break
                if maze[test_y, test_x] == 1:
                    break
                distance += 1
            noisy_distance = max(0, distance + np.random.normal(0, noise_std))
            readings.append(noisy_distance)
        return readings

###############################
# Step 2: Localization (EKF)  #
###############################

class EKF:
    def __init__(self, init_state):
        """
        Initialize the EKF with an initial state vector [x, y, theta] and covariance.
        """
        self.state = init_state  # Estimated state
        self.P = np.eye(3) * 5.0  # Initial covariance matrix

    def predict(self, control, dt):
        """
        Prediction step of the EKF.
        control: [v, omega]
        Uses the robot's motion model to predict the next state.
        """
        v, omega = control
        theta = self.state[2]

        # Motion model: differential drive
        if abs(omega) > 1e-5:
            dx = -(v/omega) * math.sin(theta) + (v/omega) * math.sin(theta + omega * dt)
            dy = (v/omega) * math.cos(theta) - (v/omega) * math.cos(theta + omega * dt)
        else:
            dx = v * math.cos(theta) * dt
            dy = v * math.sin(theta) * dt
        dtheta = omega * dt

        # Predicted state update
        self.state = self.state + np.array([dx, dy, dtheta])
        self.state[2] %= 2 * math.pi

        # Jacobian of the motion model with respect to the state (F)
        F = np.array([[1, 0, -v * math.sin(theta) * dt],
                      [0, 1,  v * math.cos(theta) * dt],
                      [0, 0, 1]])
        # Process noise covariance (Q)
        Q = np.diag([0.1, 0.1, 0.05])
        # Covariance update
        self.P = F.dot(self.P).dot(F.T) + Q

    def update(self, measurement, maze, sensor_angle):
        """
        Update step of the EKF using a measurement from a sensor.
        Here, we use the front sensor reading (sensor_angle relative to robot's heading).
        measurement: observed distance (scalar)
        maze: the environment map (for ray-casting)
        sensor_angle: relative angle of the sensor (e.g., 0 for the front sensor)
        """
        # Compute expected measurement using ray-casting from the estimated state.
        theta = self.state[2]
        angle = theta + sensor_angle
        expected_distance = 0
        while expected_distance < 100:
            test_x = int((self.state[0] + expected_distance * math.cos(angle)) // CELL_SIZE)
            test_y = int((self.state[1] + expected_distance * math.sin(angle)) // CELL_SIZE)
            if test_x < 0 or test_x >= GRID_WIDTH or test_y < 0 or test_y >= GRID_HEIGHT:
                break
            if maze[test_y, test_x] == 1:
                break
            expected_distance += 1

        # Approximate measurement function derivative using numerical differentiation.
        delta = 1e-3
        state_plus = self.state.copy()
        state_plus[2] += delta
        angle_plus = state_plus[2] + sensor_angle
        expected_distance_plus = 0
        while expected_distance_plus < 100:
            test_x = int((state_plus[0] + expected_distance_plus * math.cos(angle_plus)) // CELL_SIZE)
            test_y = int((state_plus[1] + expected_distance_plus * math.sin(angle_plus)) // CELL_SIZE)
            if test_x < 0 or test_x >= GRID_WIDTH or test_y < 0 or test_y >= GRID_HEIGHT:
                break
            if maze[test_y, test_x] == 1:
                break
            expected_distance_plus += 1

        H_theta = (expected_distance_plus - expected_distance) / delta
        H = np.array([[0, 0, H_theta]])
        R = np.array([[1.0]])  # Measurement noise covariance

        S = H.dot(self.P).dot(H.T) + R
        K = self.P.dot(H.T).dot(np.linalg.inv(S))  # Kalman gain

        innovation = measurement - expected_distance
        self.state = self.state + (K.flatten() * innovation)
        self.P = (np.eye(3) - K.dot(H)).dot(self.P)

#########################################
# Step 3: Mapping (Occupancy Grid)      #
#########################################

class OccupancyGrid:
    def __init__(self, width, height):
        """
        Initialize the occupancy grid.
        The grid is initialized with 0.5 representing unknown probability.
        """
        self.width = width
        self.height = height
        self.grid = 0.5 * np.ones((height, width))

    def update(self, robot, sensor_readings):
        """
        Update the occupancy grid using the robot's sensor readings.
        For each sensor reading, mark cells along the ray as free (0.0) and
        mark the cell where an obstacle is detected as occupied (1.0).
        """
        x, y, theta = robot.get_pose()
        for i, distance in enumerate(sensor_readings):
            sensor_angle = theta + robot.sensor_angles[i]
            # Mark cells along the ray as free
            for d in range(0, int(distance)):
                cell_x = int((x + d * math.cos(sensor_angle)) // CELL_SIZE)
                cell_y = int((y + d * math.sin(sensor_angle)) // CELL_SIZE)
                if 0 <= cell_x < self.width and 0 <= cell_y < self.height:
                    self.grid[cell_y, cell_x] = 0.0  # free
            # Mark the obstacle cell as occupied
            cell_x = int((x + distance * math.cos(sensor_angle)) // CELL_SIZE)
            cell_y = int((y + distance * math.sin(sensor_angle)) // CELL_SIZE)
            if 0 <= cell_x < self.width and 0 <= cell_y < self.height:
                self.grid[cell_y, cell_x] = 1.0  # occupied

#############################################
# Main Simulation Loop (Steps 1, 2, and 3)   #
#############################################

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Robot Simulator with EKF Localization and Mapping (Step 3)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    # Create a robot instance at a starting position.
    robot = Robot(CELL_SIZE * 1.5, CELL_SIZE * 1.5, 0)
    # Initialize the EKF with the same initial state.
    ekf = EKF(robot.get_pose().copy())
    # Initialize the occupancy grid (mapping)
    occ_grid = OccupancyGrid(GRID_WIDTH, GRID_HEIGHT)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Keyboard Controls for robot movement ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            robot.set_velocity(100, robot.angular_velocity)
        elif keys[pygame.K_DOWN]:
            robot.set_velocity(-100, robot.angular_velocity)
        else:
            robot.set_velocity(0, robot.angular_velocity)

        if keys[pygame.K_LEFT]:
            robot.angular_velocity = -2
        elif keys[pygame.K_RIGHT]:
            robot.angular_velocity = 2
        else:
            robot.angular_velocity = 0

        dt = 0.1
        control = [robot.linear_velocity, robot.angular_velocity]
        ekf.predict(control, dt)
        robot.update(dt, maze_grid)

        # Obtain sensor readings; use front sensor (index 1) for EKF update.
        sensor_readings = robot.get_sensor_readings(maze_grid)
        measurement = sensor_readings[1]
        ekf.update(measurement, maze_grid, sensor_angle=0)

        # Update the occupancy grid mapping.
        occ_grid.update(robot, sensor_readings)

        ###################
        # Rendering Phase #
        ###################
        screen.fill(WHITE)
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
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, color, rect)

        # Draw the maze boundaries on top of the occupancy grid
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if maze_grid[i, j] == 1:
                    pygame.draw.rect(screen, BLACK, rect)

        # Draw the true robot pose (blue circle with red heading)
        pygame.draw.circle(screen, BLUE, (int(robot.x), int(robot.y)), robot.radius)
        true_end_x = int(robot.x + robot.radius * math.cos(robot.theta))
        true_end_y = int(robot.y + robot.radius * math.sin(robot.theta))
        pygame.draw.line(screen, RED, (int(robot.x), int(robot.y)), (true_end_x, true_end_y), 2)

        # Draw the EKF estimated pose (yellow circle)
        est = ekf.state
        pygame.draw.circle(screen, YELLOW, (int(est[0]), int(est[1])), robot.radius // 2)
        est_end_x = int(est[0] + (robot.radius // 2) * math.cos(est[2]))
        est_end_y = int(est[1] + (robot.radius // 2) * math.sin(est[2]))
        pygame.draw.line(screen, RED, (int(est[0]), int(est[1])), (est_end_x, est_end_y), 2)

        # Optionally, display sensor readings near the robot
        for i, reading in enumerate(sensor_readings):
            angle = robot.theta + robot.sensor_angles[i]
            text = font.render(f"{reading:.0f}", True, RED)
            text_x = int(robot.x + (reading + 10) * math.cos(angle))
            text_y = int(robot.y + (reading + 10) * math.sin(angle))
            screen.blit(text, (text_x, text_y))

        # Display EKF estimated state
        est_text = font.render(f"EKF: x={est[0]:.1f}, y={est[1]:.1f}, θ={math.degrees(est[2]):.1f}°", True, RED)
        screen.blit(est_text, (10, WINDOW_HEIGHT - 30))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()