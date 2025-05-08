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
        self.fitness = []

    def add_fitness(self, fitness: (float, int)):
        self.fitness.append(fitness)

    def average_fitness(self):
        if len(self.fitness) == 0:
            return 0
        return sum(self.fitness) / len(self.fitness)

class Evolution:
    def __init__(self, pop_size: int, select_perc: float, error_range: float, mutate_perc: float):
        self.population = [Individual(RobotBrain().to(device)) for _ in range(pop_size)]
        self.select_percentage = select_perc
        self.error_range = error_range
        self.mutate_percentage = mutate_perc

    def compute_individual_fitness(self, individual, robot):
        ann_inputs = robot.get_ann_inputs()
        action = individual.brain.predict(ann_inputs)
        robot.set_velocity(action)
        # TODO: Implement the actual fitness calculation based on robot's performance
        v_l, v_r = robot.left_velocity, robot.right_velocity
        penalty = -0.5*len(robot.path_history) - robot.num_collisions if v_l == 0 and v_r == 0 else 0
        fitness = robot.get_distance_traveled() + len(robot.visible_measurements) + penalty
        individual.add_fitness(fitness)

    def select_parents(self, type='max'):
        num_parents = int(len(self.population) * self.select_percentage)
        if type == 'max':
            sorted_population = sorted(self.population, key=lambda ind: ind.average_fitness(), reverse=True)
            return sorted_population[:num_parents]
        else:
            raise ValueError("Invalid selection type. Use 'max' or 'random'.")

    def reproduce(self, parents, type='average'):
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
            if type == 'average':
                # Create child state dict by averaging parameters
                for name in p1_dict:
                    child_dict[name] = (p1_dict[name] + p2_dict[name]) / 2.0
            elif type == 'crossover':
                for name, param in p1_dict.items():
                    # Use binary mask for parameter selection (faster than multiple random calls)
                    mask = (torch.rand_like(param, dtype=torch.float) < 0.5)
                    child_dict[name] = torch.where(mask, param, p2_dict[name])
            else:
                raise ValueError("Invalid reproduction type. Use 'average' or 'crossover'.")

            # Load the parameters into the child
            child.brain.load_state_dict(child_dict)
            children.append(child)

        return children

    def mutate(self, children: list[Individual]):
        for child in children:
            if torch.rand(1) < self.mutate_percentage:
                for param in child.brain.parameters():
                    noise = torch.randn_like(param) * self.error_range
                    param.data += noise

    def create_next_generation(self, reproduction_type='crossover', num_elites=2):
        selected_parents = self.select_parents()
        children = self.reproduce(selected_parents, type=reproduction_type)
        self.mutate(children)
        # Preserve the best individuals
        children[:num_elites] = selected_parents[:num_elites]
        self.population = children