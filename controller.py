import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


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
        x = F.tanh(self.fc3(x))
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

    def compute_individual_fitness_3(self, individual, robot):
        grid_stats = robot.mapping.get_confidence_stats() if hasattr(robot, 'mapping') else {}
        confident_free = torch.tensor(grid_stats.get('confidence_ratio', 0), dtype=torch.float32)

        # Filter uncertainty
        if hasattr(robot, 'filter') and robot.filter.uncertainty_history:
            cov = robot.filter.uncertainty_history[-1]
            filter_cov = torch.tensor(cov['semi_major'] + cov['semi_minor'], dtype=torch.float32)
        else:
            filter_cov = torch.tensor(0.0)

        # Episode totals
        dust_reward = torch.tensor(robot.dust_collected(percentage=True), dtype=torch.float32)
        dist_traveled = torch.tensor(robot.get_distance_traveled(), dtype=torch.float32)
        energy_used = torch.tensor(robot.total_energy_used, dtype=torch.float32)
        collisions = torch.tensor(robot.num_collisions, dtype=torch.float32)

        # Optional: penalize being idle
        idle_penalty = torch.tensor(0.0)
        if dist_traveled.item() < 50.0:
            idle_penalty = torch.tensor(50.0)

        # Final fitness
        fitness = (  # Stronger penalty for not reaching the goal
                0.1 * dist_traveled +  # Encourage movement
                0.1 * confident_free +  # Encourage confident mapping
                10.0 * dust_reward +  # Strong reward for collecting dust
                -2.0 * collisions +  # Heavier penalty for collisions
                -0.02 * energy_used +  # Slight penalty for energy usage
                -0.05 * filter_cov +  # Penalty for being uncertain
                -idle_penalty  # Penalize being idle
        )

        individual.fitness += fitness.item()


    def compute_individual_fitness_4(self, individual, robot):
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

        idle_penalty = torch.tensor(0.0)
        if dist_traveled.item() < 50.0:
            idle_penalty = torch.tensor(50.0)

        # Dynamic weights adjustment
        w_dust = 10.0 / (1.0 + dust_reward.item())  # decay as more dust is collected
        w_confidence = 0.2 + 0.8 * confident_free.item()  # increase as confidence improves
        w_cov = -0.05 * (1.0 + filter_cov.item())  # increase penalty as uncertainty rises
        w_collisions = -2.0 * (1.0 + collisions.item())  # heavier penalty with more collisions

        # Final fitness with dynamic weights
        fitness = (
            0.1 * dist_traveled +
            w_confidence * confident_free +
            w_dust * dust_reward +
            w_collisions +
            -0.02 * energy_used +
            w_cov +
            -idle_penalty
        )

        individual.fitness += fitness.item()

    def select_parents(self, method):
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

    def reproduce(self, parents, method):
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

    def create_next_generation(self, reproduction_type='crossover', selection_type='tournament', num_elites=2):
        selected_parents = self.select_parents(method=selection_type)
        children = self.reproduce(selected_parents, method=reproduction_type)
        self.mutate(children)
        # Preserve the best individuals
        children[:num_elites] = selected_parents[:num_elites]
        self.population = children
        for individual in self.population:
            individual.reset_fitness()

    