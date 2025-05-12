import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class RobotBrain(nn.Module):
    def __init__(self):
        super(RobotBrain, self).__init__()
        # 16 inputs (12 sensor readings + robot pose + bias)
        # 6 outputs (based on the number of actions)
        self.fc1 = nn.Linear(16, 60)
        self.fc2 = nn.Linear(60, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
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

    def add_fitness(self, fitness: (float, int)):
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

    def compute_individual_fitness(self, individual, robot):
        grid_stats = robot.mapping.get_confidence_stats() if hasattr(robot, 'mapping') else None
        filter_covariance = robot.filter.uncertainty_history[-1]['semi_major'] + robot.filter.uncertainty_history[-1][
            'semi_minor'] if hasattr(robot, 'filter') else None

        x = torch.tensor([
            robot.get_distance_traveled(),
            len(robot.visible_measurements),
            sum([r for r, _ in robot.sensor_readings]) - len(robot.sensor_readings) * robot.sensor_range,
            grid_stats['confident_free'] if grid_stats else 0,
            -filter_covariance if filter_covariance else 0,
            -robot.num_collisions
        ], device=device)

        a = torch.tensor([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001], device=device)

        fitness_value = torch.dot(x, a).item()
        individual.add_fitness(fitness_value)

    def select_parents(self, method='max'):
        num_parents = int(len(self.population) * self.select_percentage)
        if method == 'max':
            sorted_population = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
            return sorted_population[:num_parents]
        else:
            raise ValueError("Invalid selection type. Use 'max' or 'random'.")

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