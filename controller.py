import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class RobotBrain(nn.Module):
    def __init__(self):
        super(RobotBrain, self).__init__()
        # 16 inputs (12 sensor readings + robot pose + bias)
        # 6 outputs (based on the number of actions)
        self.fc1 = nn.Linear(16, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            output = self.forward(state_tensor)
            exp_values = torch.exp(output - torch.max(output))
            probabilities = exp_values / torch.sum(exp_values)
            action = torch.multinomial(probabilities, 1).item()
            return action

class Individual:
    def __init__(self, brain: RobotBrain):
        self.brain = brain
        self.fitness = 0.0

    def add_fitness(self, fitness):
        self.fitness += fitness

    def reset_fitness(self):
        self.fitness = 0.0

class Evolution:
    def __init__(self, pop_size: int, select_perc: float, error_range: float, mutate_perc: float):
        self.population = [Individual(RobotBrain().to(device)) for _ in range(pop_size)]
        self.select_percentage = select_perc
        self.error_range = error_range
        self.mutate_percentage = mutate_perc
        self.create_diverse_population()

    def create_diverse_population(self):
        """Create a diverse initial population using multiple initialization strategies."""
        for i, individual in enumerate(self.population):
            # Apply different initialization strategies based on index
            if i % 3 == 0:
                # Strategy 1: Xavier/Glorot initialization
                for name, param in individual.brain.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
            elif i % 3 == 1:
                # Strategy 2: Kaiming/He initialization
                for name, param in individual.brain.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param.data, nonlinearity='relu')
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
            else:
                # Strategy 3: Random noise injection with larger variance
                for param in individual.brain.parameters():
                    noise = torch.randn_like(param) * (self.error_range * 2.0)
                    param.data += noise

    def compute_individual_fitness_3(self, individual, robot):
        # Retrieve metrics
        dust_collected = robot.dust_collected(percentage=True)
        distance_traveled = robot.get_distance_traveled()
        collisions = robot.num_collisions
        energy_used = robot.total_energy_used
        confidence_ratio = robot.mapping.get_confidence_stats().get('confidence_ratio', 0) if hasattr(robot, 'mapping') else 0
        uncertainty = robot.filter.uncertainty_history[-1]['semi_major'] + robot.filter.uncertainty_history[-1][
            'semi_minor'] if hasattr(robot, 'filter') else 0

        # Normalize metrics
        dust_norm = dust_collected
        distance_norm = distance_traveled / 1000 # Assume max 1000 distance
        energy_norm = energy_used / 1000  # Assume max 1000 energy
        collisions_norm = collisions / 10  # Assume max 10 collisions
        uncertainty_norm = uncertainty / 100  # Assume max 100 uncertainty

        # Compute fitness
        fitness = (
                10 * dust_norm +  # Reward for dust collection
                0.1 * distance_norm +  # Reward for exploration
                0.2 * confidence_ratio -  # Reward for mapping confidence
                2 * collisions_norm -  # Penalty for collisions
                0.05 * energy_norm -  # Penalty for energy usage
                0.1 * uncertainty_norm  # Penalty for uncertainty
        )

        individual.add_fitness(fitness)

    def compute_individual_fitness_2(self, individual, robot):
        """
        Fitness combines:
         - dust collected (reward),
         - distance traveled (exploration),
         - time steps / coverage (visiting new cells),
         - mapping confidence ratio,
         - localization uncertainty (penalty),
         - collisions (penalty).
        """
        # raw metrics
        dust       = robot.dust_count
        distance   = robot.get_distance_traveled()
        coverage   = len(robot.path_history)                    
        grid_stats = robot.mapping.get_confidence_stats() \
                       if hasattr(robot, 'mapping') else None
        confidence = grid_stats['confidence_ratio'] if grid_stats else 0.0
        cov        = (robot.filter.uncertainty_history[-1]['semi_major']
                      + robot.filter.uncertainty_history[-1]['semi_minor']) \
                      if hasattr(robot, 'filter') else 0.0
        collisions = robot.num_collisions

        # feature vector
        x = torch.tensor([ 
            dust,
            distance,
            coverage,
            confidence,
            -cov,       # less uncertainty is better
            -collisions # fewer collisions is better
        ], device=device, dtype=torch.float32)

        
        a = torch.tensor([
            1.0,    # dust
            0.01,   # distance
            0.01,  # coverage 
            0.1,    # mapping confidence
            0.1,    # localization
            0.5     # safety
        ], device=device, dtype=torch.float32)

        fitness_value = torch.dot(x, a).item()
        individual.add_fitness(fitness_value)

    def select_parents(self, method='tournament'):
        num_parents = int(len(self.population) * self.select_percentage)
        if method == 'max':
            sorted_population = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
            return sorted_population[:num_parents]
        elif method == 'random':
            selected_indices = torch.randint(0, len(self.population), (num_parents,))
            return [self.population[i] for i in selected_indices.tolist()]
        elif method == 'tournament':
            selected_parents = []
            for _ in range(num_parents):
                # Randomly select a subset of individuals for the tournament
                tournament_size = int(len(self.population) * 0.2)
                tournament_indices = torch.randint(0, len(self.population), (tournament_size,))
                tournament_individuals = [self.population[i] for i in tournament_indices.tolist()]
                # Select the best individual from the tournament
                best_individual = max(tournament_individuals, key=lambda ind: ind.fitness)
                selected_parents.append(best_individual)
            return selected_parents
        else:
            raise ValueError("Invalid selection type. Use 'max', 'random' or 'tournament'.")

    def reproduce(self, parents, method='crossover'):
        num_children = len(self.population)
        children = []

        # Generate two random parent indices for each child
        parent_indices = torch.randint(0, len(parents), (num_children, 2))

        # Create child models
        child_models = [Individual(RobotBrain().to(device)) for _ in range(num_children)]

        # Process each child
        for i in range(num_children):
            p1_idx, p2_idx = parent_indices[i].tolist()
            child = child_models[i]

            # Get parent state dictionaries
            p1_dict = parents[p1_idx].brain.state_dict()
            p2_dict = parents[p2_idx].brain.state_dict()

            child_dict = {}
            if method == 'average':
                # Create child state dict by averaging parameters
                for name in p1_dict:
                    child_dict[name] = (p1_dict[name] + p2_dict[name]) / 2.0
                child.brain.load_state_dict(child_dict)
            elif method == 'crossover':
                for name, param in p1_dict.items():
                    # Use binary mask for parameter selection (faster than multiple random calls)
                    mask = (torch.rand_like(param, dtype=torch.float) < 0.5)
                    child_dict[name] = torch.where(mask, param, p2_dict[name])
                child.brain.load_state_dict(child_dict)
            else:
                raise ValueError("Invalid reproduction type. Use 'average' or 'crossover'.")
            children.append(child)

        return children

    def mutate(self, children: list[Individual]):
        for child in children:
            if torch.rand(1) < self.mutate_percentage:
                for param in child.brain.parameters():
                    noise = torch.randn_like(param) * self.error_range
                    param.data += noise

    def create_next_generation(self, reproduction_type='crossover', selection_type='max', num_elites=2):
        selected_parents = self.select_parents(method=selection_type)
        children = self.reproduce(selected_parents, method=reproduction_type)
        self.mutate(children)
        # Preserve the best individuals
        children[:num_elites] = selected_parents[:num_elites]
        self.population = children
        for individual in self.population:
            individual.reset_fitness()

    def compute_individual_fitness_4(self, individual, robot):
        import torch
        from config import Config

        # Helper function for min-max normalization
        def normalize(value, min_val, max_val):
            EPS = 1e-6
            return (value - min_val) / (max_val - min_val + EPS)

        # Retrieve relevant stats
        grid_stats = robot.mapping.get_confidence_stats() if hasattr(robot, 'mapping') else {}
        confident_free = torch.tensor(grid_stats.get('confidence_ratio', 0), dtype=torch.float32)

        if hasattr(robot, 'filter') and robot.filter.uncertainty_history:
            cov = robot.filter.uncertainty_history[-1]
            filter_cov = torch.tensor(cov['semi_major'] + cov['semi_minor'], dtype=torch.float32)
        else:
            filter_cov = torch.tensor(0.0)

        dust_reward = torch.tensor(robot.dust_collected(), dtype=torch.float32)
        dist_traveled = torch.tensor(robot.get_distance_traveled(), dtype=torch.float32)
        energy_used = torch.tensor(robot.total_energy_used, dtype=torch.float32)
        collisions = torch.tensor(robot.num_collisions, dtype=torch.float32)

        # Idle penalty normalization
        idle_raw = 50.0 if dist_traveled.item() < 50.0 else 0.0
        idle_norm = normalize(idle_raw, 0, 50.0)

        # Normalize metrics (use conservative fixed ranges)
        dust_norm     = normalize(dust_reward.item(),      0, 20)   # Max 20 dust
        dist_norm     = normalize(dist_traveled.item(),    0, 100)  # Max 100 distance
        conf_norm     = normalize(confident_free.item(),   0, 1)    # Already a ratio
        cov_norm      = normalize(filter_cov.item(),       0, 100)  # Max 100 for uncertainty
        coll_norm     = normalize(collisions.item(),       0, 10)   # Max 10 collisions
        energy_norm   = normalize(energy_used.item(),      0, 1000) # Max 1000 energy

        # Dynamic weights based on normalized values
        w_dust       = 10.0 / (1.0 + dust_norm)
        w_confidence = 0.2 + 0.8 * conf_norm
        w_cov        = -0.05 * (1.0 + cov_norm)
        w_collisions = -2.0 * (1.0 + coll_norm)

        # Final fitness with dynamic weights
        fitness = (
            0.1 * dist_norm +
            w_confidence * conf_norm +
            w_dust * dust_norm +
            w_collisions +
            -0.02 * energy_norm +
            w_cov +
            -idle_norm
        )

        individual.fitness += fitness