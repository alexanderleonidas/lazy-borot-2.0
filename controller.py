import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
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
            probabilities = F.softmax(output, dim=0)
            action = torch.multinomial(probabilities, 1).item()
            return action

    def predict_with_temperature(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            output = self.forward(state_tensor)
            temperature = 0.5  # Adjust for exploration/exploitation balance
            scaled_output = output / temperature
            probabilities = F.softmax(scaled_output, dim=0)
            action = torch.multinomial(probabilities, 1).item()
            return action

class RobotBrain(nn.Module):
    def __init__(self):
        super(RobotBrain, self).__init__()
        self.input_size = 16
        self.hidden_size = 64
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 2)  # Output left/right velocity
        self.hidden = None
        self.max_speed = 100 # Max speed of the robot, must correspond to the robot's max speed in robot.py

    def reset_hidden(self):
        self.hidden = None

    def forward(self, x):
        # Add sequence dimension
        x = x.view(1, 1, -1)
        if self.hidden is None:
            out, self.hidden = self.lstm(x)
        else:
            out, self.hidden = self.lstm(x, self.hidden)
        x = torch.tanh(self.fc(out[:, -1, :]))  # Use the last output of the LSTM
        return x.squeeze(0)

    def continuous_predict(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            output = self.forward(state_tensor)
            # Output direct wheel velocities instead of actions
            left_velocity = torch.tanh(output[0]) * self.max_speed
            right_velocity = torch.tanh(output[1]) * self.max_speed
            return (left_velocity.item(), right_velocity.item())

    # Use this only if the Action class is used for robot control
    def discrete_predict(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            output = self.forward(state_tensor)
            temperature = 1.0  # Adjust for exploration/exploitation balance
            scaled_output = output / temperature
            probabilities = F.softmax(scaled_output, dim=0)
            action = torch.multinomial(probabilities, 1).item()
            return action

class Individual:
    def __init__(self, brain: RobotBrain, multi_objective=False):
        self.brain = brain
        self.multi_objective = multi_objective
        if multi_objective:
            self.objectives = []  # Store multiple objective values
        else:
            self._fitness = 0.0


    def set_objectives(self, objective_values: (list[float], float)):
        if self.multi_objective:
            self.objectives = [float(ov) for ov in objective_values] # Ensure float
        else:
            self._fitness = float(objective_values)

    def reset_objectives(self):
        if self.multi_objective:
            self.objectives = []
        else:
            self._fitness = 0.0

    def fitness(self):
        if self.multi_objective:
            return sum(self.objectives)
        else:
            return self._fitness


class Evolution:
    def __init__(self, pop_size: int, select_perc: float, error_range: float, mutate_perc: float, multi_objective=False, rnn=False):
        self.population = [Individual(RobotBrain().to(device) if rnn else ANN().to(device), multi_objective=multi_objective) for _ in range(pop_size)]
        self.select_percentage = select_perc
        self.error_range = error_range
        self.mutate_percentage = mutate_perc
        self.create_diverse_population()
        self.multi_objective = multi_objective
        self.objective_names_map = [
            "Dust Collected(%)",  # Higher is better
            "Distance Traveled",  # Higher is better
            "Negative Collisions",  # Higher is better (means fewer collisions)
            "Negative Energy Used"  # Higher is better (means less energy used)
        ]

    def create_diverse_population(self):
        """Create a diverse initial population using multiple initialization strategies."""
        for i, individual in enumerate(self.population):
            # Apply different initialization strategies based on index
            if i % 3 == 1:
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

    def compute_individual_fitness(self, individual, robot, weights):
        dust_reward = torch.tensor(robot.dust_collected(), dtype=torch.float32)
        dist_traveled = torch.tensor(robot.get_distance_traveled(), dtype=torch.float32)
        energy_used = torch.tensor(robot.total_energy_used, dtype=torch.float32)
        collisions = torch.tensor(robot.num_collisions, dtype=torch.float32)

        idle_raw = 50.0 if dist_traveled.item() < 50.0 else 0.0

        fitness = (weights[0] * dust_reward + weights[1] * dist_traveled + weights[2] * energy_used + weights[
            3] * collisions +
                   weights[4] * idle_raw)

        individual.fitness += fitness

    def compute_individual_objectives(self, individual, robot):
        """
        Primary Multi-Objective fitness calculation.
        Objectives (all to be maximized):
        1. Dust Collected
        2. Distance Traveled
        3. - Collisions (negative of collision count)
        4. - Energy Used (negative of energy used)
        """
        dust_collected = robot.dust_collected() # Assuming this returns a numeric value
        distance_traveled = robot.get_distance_traveled()
        collisions = robot.num_collisions
        energy_used = robot.total_energy_used

        # Calculate path efficiency to discourage circles
        if len(robot.path_history) > 10:
            # Compare the actual path to the direct distance between start and end
            start_x, start_y = robot.path_history[0]
            end_x, end_y = robot.path_history[-1]
            direct_distance = torch.sqrt(torch.tensor((end_x - start_x) ** 2 + (end_y - start_y) ** 2))
            path_efficiency = direct_distance / max(1.0, distance_traveled)
        else:
            path_efficiency = 0.5  # Default for short paths

        objectives = [
            dust_collected,
            distance_traveled*path_efficiency,
            -float(collisions),      # Negate for maximization
            -float(energy_used)      # Negate for maximization
        ]
        individual.set_objectives(objectives)

    def _dominates(self, objectives1: list[float], objectives2: list[float]) -> bool:
        if not objectives1 or not objectives2 or len(objectives1) != len(objectives2):
            return False

        # Check if objectives1 is strictly better in at least one objective
        # and not worse in any other.
        better_in_at_least_one = False
        for o1, o2 in zip(objectives1, objectives2):
            if o1 < o2:  # objectives1 is worse in this objective
                return False
            if o1 > o2:  # objectives1 is better in this objective
                better_in_at_least_one = True
        return better_in_at_least_one

    def get_non_dominated_solutions(self, population_subset=None) -> list[Individual]:
        if population_subset is None:
            population_subset = self.population

        # Filter out individuals that haven't been evaluated (no objectives)
        evaluated_population = [ind for ind in population_subset if ind.objectives]
        if not evaluated_population:
            return []

        non_dominated_individuals = []
        for i, p_ind in enumerate(evaluated_population):
            is_dominated = False
            for j, q_ind in enumerate(evaluated_population):
                if i == j:
                    continue
                if self._dominates(q_ind.objectives, p_ind.objectives):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated_individuals.append(p_ind)
        return non_dominated_individuals

    def plot_pareto_front(self, objective_indices: list[int], population_subset=None, title_suffix=""):
        if not (2 <= len(objective_indices) <= 3):
            raise ValueError("objective_indices must contain 2 or 3 integer indices for plotting.")
        if not all(isinstance(idx, int) and 0 <= idx < len(self.objective_names_map) for idx in objective_indices):
            raise ValueError(
                f"objective_indices must be valid indices for self.objective_names_map (0 to {len(self.objective_names_map) - 1}).")

        if population_subset is None:
            population_subset = self.population

        non_dominated_sols = self.get_non_dominated_solutions(population_subset)

        if not non_dominated_sols:
            print("No non-dominated solutions found to plot.")
            return

        valid_solutions_for_plot = []
        for sol in non_dominated_sols:
            # All non_dominated_sols should have objectives by definition of get_non_dominated_solutions
            # We just need to ensure the indices are valid for the number of objectives they have.
            if all(idx < len(sol.objectives) for idx in objective_indices):
                valid_solutions_for_plot.append(sol)
            else:
                print(f"Warning: A non-dominated solution was skipped for plotting. "
                      f"It has {len(sol.objectives)} objective(s), but plotting requires up to index {max(objective_indices)}.")

        if not valid_solutions_for_plot:
            print("No valid non-dominated solutions for plotting after filtering by objective indices.")
            return

        plot_pareto_solutions(
            solutions_to_plot=valid_solutions_for_plot,
            objective_indices=objective_indices,
            all_objective_names=self.objective_names_map,
            title_suffix=title_suffix
        )

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
        elif method == 'tournament_mo':
            selected_parents = []
            # Binary tournament selection using Pareto dominance
            tournament_size = 2  # Standard binary tournament
            for _ in range(num_parents):
                competitors = random.sample(self.population, tournament_size)

                p1 = competitors[0]
                if tournament_size == 1:  # Edge case, unlikely but handle
                    selected_parents.append(p1)
                    continue
                p2 = competitors[1]

                if self._dominates(p1.objectives, p2.objectives):
                    selected_parents.append(p1)
                elif self._dominates(p2.objectives, p1.objectives):
                    selected_parents.append(p2)
                else:  # Non-dominated or identical, pick one randomly
                    selected_parents.append(random.choice(competitors))
            return selected_parents
        else:
            raise ValueError("Invalid selection type. Use 'max', 'random', 'tournament' or 'tournament_mo'.")

    def reproduce(self, parents, method='crossover'):
        num_children = len(self.population)
        children = []

        # Generate two random parent indices for each child
        parent_indices = torch.randint(0, len(parents), (num_children, 2))

        # Create child models
        child_models = [Individual(RobotBrain().to(device), multi_objective=self.multi_objective) for _ in range(num_children)]

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

    def create_next_generation_mo(self, reproduction_method='crossover', selection_method='tournament_mo', num_elites=2):
        # 1. Elitism: Select elites from the current population based on non-domination
        elites = []
        if num_elites > 0:
            non_dominated_current_pop = self.get_non_dominated_solutions(self.population)
            if non_dominated_current_pop:
                # If more non-dominated solutions than num_elites, shuffle and pick
                if len(non_dominated_current_pop) > num_elites:
                    random.shuffle(non_dominated_current_pop)
                elites = non_dominated_current_pop[:num_elites]

        # 2. Parent Selection
        parents = self.select_parents(method=selection_method)

        # 3. Reproduction
        num_offspring = len(self.population) - len(elites)  # Create offspring to fill the rest
        if num_offspring <= 0:  # Edge case: elites fill the whole population
            children = []
        elif not parents:  # No parents selected, perhaps the population was too small or unevaluated
            print(
                f"Warning: No parents selected. Creating {num_offspring} new random individuals instead of offspring.")
            children = [Individual(RobotBrain().to(device), multi_objective=self.multi_objective) for _ in range(num_offspring)]
            # Optionally re-initialize these new random individuals
            # for child in children: self.create_diverse_population_single(child) # (if such a method existed)
        else:
            # Adjust `reproduce` to create a specific number of children
            # For now, reproduce still makes len(self.population) children, so we'll take a slice.
            all_potential_children = self.reproduce(parents, method=reproduction_method)
            children = all_potential_children[:num_offspring]

        # 4. Mutation
        self.mutate(children)

        # 5. Form new population
        self.population = elites + children

        # Reset objectives for the new generation (elites retain theirs until re-evaluated)
        for individual in self.population:
            if individual not in elites:  # New children need their objectives reset
                individual.reset_objectives()
            # Elites will be re-evaluated with the rest of the population in the next cycle.