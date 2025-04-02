import numpy as np
import random
from config import Config

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
            target = np.array([Config.WINDOW_WIDTH/2, Config.WINDOW_HEIGHT/2])
            for _ in range(self.num_particles):
                pos = target + np.random.randn(2) * 10
                self.particles.append(Particle(pos))
        else:
            # Randomly initialize particles based on frontier cells.
            for _ in range(self.num_particles):
                frontier = random.choice(frontiers)
                pos = np.array([frontier[0]*Config.CELL_SIZE + Config.CELL_SIZE/2, frontier[1]*Config.CELL_SIZE + Config.CELL_SIZE/2])
                pos += np.random.randn(2) * 5  # add a bit of randomness
                self.particles.append(Particle(pos))

    def cost_function(self, candidate, robot_pos):
        grid_x = int(candidate[0] // Config.CELL_SIZE)
        grid_y = int(candidate[1] // Config.CELL_SIZE)
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