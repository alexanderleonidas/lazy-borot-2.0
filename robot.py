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
CELL_SIZE = 48 # pixels per cell

# Pygame window size
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

# Define colors
WHITE  = (255, 255, 255)
BLACK  = (  0,   0,   0)
GRAY   = (120, 120, 120)
BLUE   = (  0,   0, 255)
GREEN  = (  0, 255,   0)
RED    = (255,   0,   0)
YELLOW = (255, 255,   0)

#########################
# Milestone 1: Simulator #
#########################

class Robot:
    def __init__(self, x, y, theta):
        # Real (ground truth) pose
        self.x = x  
        self.y = y
        self.theta = theta  # radians

        self.radius = 10  # for drawing the robot
        self.linear_velocity = 0.0  # pixels per second
        self.angular_velocity = 0.0  # radians per second
        self.wheel_base = 20  # for differential drive simulation

        # Sensor configuration (simulating 3 sensors; can be extended to 12)
        self.sensor_range = 100  # max range in pixels
        self.sensor_angles = [-math.pi/4, 0, math.pi/4]  # relative sensor directions

    def update(self, dt, maze):
        # Proposed new position (for collision checking)
        new_x = self.x + self.linear_velocity * math.cos(self.theta) * dt
        new_y = self.y + self.linear_velocity * math.sin(self.theta) * dt
        
        # Check for collision with maze walls
        cell_x = int(new_x // CELL_SIZE)
        cell_y = int(new_y // CELL_SIZE)
        if (cell_x < 0 or cell_x >= GRID_WIDTH or 
            cell_y < 0 or cell_y >= GRID_HEIGHT or 
            maze[cell_y, cell_x] == 1):
            # Collision detected: halt forward movement, allow rotation (simulate sliding)
            self.linear_velocity = 0
        else:
            self.x = new_x
            self.y = new_y

        # Always update orientation
        self.theta += self.angular_velocity * dt
        self.theta %= 2 * math.pi

    def set_velocity(self, linear, angular):
        self.linear_velocity = linear
        self.angular_velocity = angular

    def get_pose(self):
        return np.array([self.x, self.y, self.theta])

    def get_sensor_readings(self, maze):
        """
        Simulate sensor readings with simple ray-casting.
        A small random noise is added to simulate realistic measurements.
        Returns a list of distances (one per sensor).
        """
        readings = []
        noise_std = 1.0  # standard deviation for noise
        for angle in self.sensor_angles:
            sensor_angle = self.theta + angle
            distance = 0
            while distance < self.sensor_range:
                test_x = int((self.x + distance * math.cos(sensor_angle)) // CELL_SIZE)
                test_y = int((self.y + distance * math.sin(sensor_angle)) // CELL_SIZE)
                if test_x < 0 or test_x >= GRID_WIDTH or test_y < 0 or test_y >= GRID_HEIGHT:
                    break
                if maze[test_y, test_x] == 1:  # hit a wall
                    break
                distance += 1
            # Add Gaussian noise to the sensor reading
            noisy_distance = max(0, distance + np.random.normal(0, noise_std))
            readings.append(noisy_distance)
        return readings

#########################################
# Milestone 2: Localization with EKF    #
#########################################

class EKF:
    def __init__(self, init_state):
        # State: [x, y, theta]
        self.state = init_state
        self.cov = np.eye(3) * 5.0  # initial covariance

    def predict(self, control, dt):
        v, omega = control
        theta = self.state[2]
        # Motion model prediction (differential drive)
        if abs(omega) > 1e-5:
            dx = -(v/omega) * math.sin(theta) + (v/omega) * math.sin(theta + omega*dt)
            dy = (v/omega) * math.cos(theta) - (v/omega) * math.cos(theta + omega*dt)
        else:
            dx = v * math.cos(theta) * dt
            dy = v * math.sin(theta) * dt
        dtheta = omega * dt
        self.state = self.state + np.array([dx, dy, dtheta])

        # Jacobian approximation of the motion model
        Fx = np.array([[1, 0, -v/omega * math.cos(theta) + v/omega * math.cos(theta + omega*dt)],
                       [0, 1, -v/omega * math.sin(theta) + v/omega * math.sin(theta + omega*dt)],
                       [0, 0, 1]])
        Q = np.diag([0.1, 0.1, 0.05])
        self.cov = Fx.dot(self.cov).dot(Fx.T) + Q

    def update(self, measurements, sensor_angles, maze, robot_pos):
        """
        For simplicity, we update using the front sensor (sensor_angles[1]).
        We compute the expected measurement by ray-casting from the predicted state.
        """
        theta = self.state[2]
        sensor_angle = theta + sensor_angles[1]
        expected_distance = 0
        while expected_distance < 100:
            test_x = int((self.state[0] + expected_distance * math.cos(sensor_angle)) // CELL_SIZE)
            test_y = int((self.state[1] + expected_distance * math.sin(sensor_angle)) // CELL_SIZE)
            if test_x < 0 or test_x >= GRID_WIDTH or test_y < 0 or test_y >= GRID_HEIGHT:
                break
            if maze[test_y, test_x] == 1:
                break
            expected_distance += 1

        z = np.array([measurements[1]])  # actual reading from the front sensor
        h = np.array([expected_distance])

        # Approximate measurement Jacobian (only with respect to theta)
        H = np.array([[0, 0, 0]])
        delta = 1e-3
        state_plus = self.state.copy()
        state_plus[2] += delta
        sensor_angle_plus = state_plus[2] + sensor_angles[1]
        expected_distance_plus = 0
        while expected_distance_plus < 100:
            test_x = int((state_plus[0] + expected_distance_plus * math.cos(sensor_angle_plus)) // CELL_SIZE)
            test_y = int((state_plus[1] + expected_distance_plus * math.sin(sensor_angle_plus)) // CELL_SIZE)
            if test_x < 0 or test_x >= GRID_WIDTH or test_y < 0 or test_y >= GRID_HEIGHT:
                break
            if maze[test_y, test_x] == 1:
                break
            expected_distance_plus += 1
        H[0,2] = (expected_distance_plus - expected_distance) / delta

        R = np.array([[1.0]])  # measurement noise covariance
        S = H.dot(self.cov).dot(H.T) + R
        K = self.cov.dot(H.T).dot(np.linalg.inv(S))
        innovation = z - h
        self.state = self.state + (K.dot(innovation)).flatten()
        self.cov = (np.eye(3) - K.dot(H)).dot(self.cov)

#############################################
# Milestone 3: Incremental Occupancy Mapping  #
#############################################

class OccupancyGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Initialize grid with unknown probability (0.5)
        self.grid = 0.5 * np.ones((height, width))

    def update(self, robot, sensor_readings):
        x, y, theta = robot.get_pose()
        for i, distance in enumerate(sensor_readings):
            sensor_angle = theta + robot.sensor_angles[i]
            # Mark cells along the ray as free
            for d in range(0, int(distance)):
                cell_x = int((x + d * math.cos(sensor_angle)) // CELL_SIZE)
                cell_y = int((y + d * math.sin(sensor_angle)) // CELL_SIZE)
                if 0 <= cell_x < self.width and 0 <= cell_y < self.height:
                    self.grid[cell_y, cell_x] = 0.0  # free space
            # Mark the cell where obstacle is detected as occupied
            cell_x = int((x + distance * math.cos(sensor_angle)) // CELL_SIZE)
            cell_y = int((y + distance * math.sin(sensor_angle)) // CELL_SIZE)
            if 0 <= cell_x < self.width and 0 <= cell_y < self.height:
                self.grid[cell_y, cell_x] = 1.0  # occupied

    def get_frontiers(self):
        """
        Identify frontier cells (free cells adjacent to unknown cells).
        """
        frontiers = []
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                if self.grid[i, j] == 0.0:  # free cell
                    # Check surrounding 3x3 neighborhood for unknown cells (0.5)
                    neighbors = self.grid[i-1:i+2, j-1:j+2]
                    if np.any(neighbors == 0.5):
                        frontiers.append((j, i))
        return frontiers

#############################################
# Milestone 4: PSO for Navigation Planning  #
#############################################

class Particle:
    def __init__(self, position):
        # Particle represents a candidate target point (x,y)
        self.position = position.copy()
        self.velocity = np.array([0.0, 0.0])
        self.best_position = position.copy()
        self.best_cost = float('inf')

class PSOPlanner:
    def __init__(self, occupancy_grid, num_particles=30, iterations=20):
        self.occupancy_grid = occupancy_grid
        self.num_particles = num_particles
        self.iterations = iterations
        self.particles = []

    def initialize_particles(self, start, frontiers):
        self.particles = []
        if len(frontiers) == 0:
            # If no frontiers are found, target the center of the map.
            target = np.array([WINDOW_WIDTH/2, WINDOW_HEIGHT/2])
            for _ in range(self.num_particles):
                pos = target + np.random.randn(2) * 10
                self.particles.append(Particle(pos))
        else:
            # Randomly initialize particles based on frontier cells.
            for _ in range(self.num_particles):
                frontier = random.choice(frontiers)
                pos = np.array([frontier[0]*CELL_SIZE + CELL_SIZE/2, frontier[1]*CELL_SIZE + CELL_SIZE/2])
                pos += np.random.randn(2) * 5  # add a bit of randomness
                self.particles.append(Particle(pos))

    def cost_function(self, candidate, robot_pos):
        grid_x = int(candidate[0] // CELL_SIZE)
        grid_y = int(candidate[1] // CELL_SIZE)
        occ_cost = 0.0
        if 0 <= grid_x < self.occupancy_grid.width and 0 <= grid_y < self.occupancy_grid.height:
            occ_cost = self.occupancy_grid.grid[grid_y, grid_x]
        dist_cost = np.linalg.norm(candidate - robot_pos)
        return dist_cost + 7 * occ_cost  # reduced multiplier for occupancy cost

    def plan(self, robot_pos):
        frontiers = self.occupancy_grid.get_frontiers()
        self.initialize_particles(robot_pos, frontiers)
        global_best_position = None
        global_best_cost = float('inf')
        # PSO parameters
        w = 0.5  # inertia weight
        c1 = 1.0  # cognitive coefficient
        c2 = 1.0  # social coefficient
        for _ in range(self.iterations):
            for particle in self.particles:
                cost = self.cost_function(particle.position, robot_pos)
                if cost < particle.best_cost:
                    particle.best_cost = cost
                    particle.best_position = particle.position.copy()
                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_position = particle.position.copy()
            for particle in self.particles:
                r1 = random.random()
                r2 = random.random()
                cognitive = c1 * r1 * (particle.best_position - particle.position)
                social = c2 * r2 * (global_best_position - particle.position)
                particle.velocity = w * particle.velocity + cognitive + social
                particle.position = particle.position + particle.velocity
        return global_best_position

#############################################
# Main Simulation Loop and Integration      #
#############################################

def main():
    pygame.init()
    font = pygame.font.SysFont(None, 24)  # for rendering text
    # Define the Search and Rescue theme's starting point and safe zone
    starting_point = np.array([CELL_SIZE * 1.5, CELL_SIZE * 1.5])
    safe_zone = np.array([540, 460])  # designated safe zone (end goal)
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Autonomous Maze-Navigating Robot using PSO")
    clock = pygame.time.Clock()

    # Initialize the robot at the starting point
    start_x = starting_point[0]
    start_y = starting_point[1]
    robot = Robot(start_x, start_y, 0)

    # Initialize EKF with the same starting state
    ekf = EKF(np.array([start_x, start_y, 0]))

    # Create occupancy grid map
    occ_grid = OccupancyGrid(GRID_WIDTH, GRID_HEIGHT)

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
        sensor_readings = robot.get_sensor_readings(maze_grid)
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
        ekf.update(sensor_readings, robot.sensor_angles, maze_grid, robot_pos)

        # Update robot’s pose with collision handling
        robot.update(dt, maze_grid)
        # Check if the robot has reached the safe zone (within 15 pixels)
        if np.linalg.norm(np.array([robot.x, robot.y]) - safe_zone) < 15:
            print("Safe zone reached. Mission complete!")
            running = False

        ###################
        # Rendering Phase #
        ###################
        screen.fill(WHITE)
        # Draw maze cells
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if maze_grid[i, j] == 1:
                    pygame.draw.rect(screen, BLACK, rect)
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
        pygame.draw.circle(screen, BLUE, (int(robot.x), int(robot.y)), robot.radius)
        end_x = int(robot.x + robot.radius * math.cos(robot.theta))
        end_y = int(robot.y + robot.radius * math.sin(robot.theta))
        pygame.draw.line(screen, RED, (int(robot.x), int(robot.y)), (end_x, end_y), 2)

        # Draw sensor rays in green.
        for i, angle in enumerate(robot.sensor_angles):
            sensor_angle = robot.theta + angle
            ray_end_x = int(robot.x + sensor_readings[i] * math.cos(sensor_angle))
            ray_end_y = int(robot.y + sensor_readings[i] * math.sin(sensor_angle))
            pygame.draw.line(screen, GREEN, (int(robot.x), int(robot.y)), (ray_end_x, ray_end_y), 1)

        # Draw the current target as a yellow circle.
        if target is not None:
            pygame.draw.circle(screen, YELLOW, (int(target[0]), int(target[1])), 5)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()