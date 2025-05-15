from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
import csv
import json
from config import Config
import torch


def save_run(run_id, robot, time_step=None, maze=None, filter_instance=None, filename="robot_pose_data.csv"):
    filename = f"results/{run_id}_{filename}"
    file_exists = os.path.isfile(filename)

    # Extract true pose
    true_pose = robot.get_pose()  # (x, y, theta)

    # Extract filter data if available
    if filter_instance:
        filter_pose = filter_instance.belief_history[-1].tolist()
        uncertainty = filter_instance.uncertainty_history[-1]  # dict: {'major': val, 'minor': val, 'angle': val}
    else:
        filter_pose = [None, None, None]
        uncertainty = {'semi_major': None, 'semi_minor': None, 'angle_rad': None}

    # Prepare row
    row = {
        "time_step": time_step,
        "true_x": true_pose[0],
        "true_y": true_pose[1],
        "true_theta": true_pose[2],
        "filter_x": filter_pose[0],
        "filter_y": filter_pose[1],
        "filter_theta": filter_pose[2],
        "uncertainty_major": uncertainty['semi_major'],
        "uncertainty_minor": uncertainty['semi_minor'],
        "uncertainty_angle": uncertainty['angle_rad']
    }

    # Write row to CSV
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # Save maze if present
    if maze is not None:
        maze_filename = filename.replace(".csv", "_maze.json")
        with open(maze_filename, 'w') as f:
            json.dump(maze.tolist(), f, indent=4)


# --- Helper function to plot fitness ---

def plot_fitness_progress(avg_fitness=None, best_fitness=None, run_id=None, generations_to_plot=None):
    """
    If run_id is provided, loads fitness data from the corresponding file and plots it.
    Otherwise, plots the provided avg_fitness and best_fitness lists.
    """
    if run_id is not None:
        # Load fitness data from file
        filename = f"results/{run_id}_training_fitness_log.csv"
        if not os.path.isfile(filename):
            print(f"Fitness log file not found: {filename}")
            return
        avg_fitness, best_fitness = [], []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                avg_fitness.append(float(row["average_fitness"]))
                best_fitness.append(float(row["best_fitness"]))
        generations_to_plot = len(avg_fitness)
    elif avg_fitness is not None and best_fitness is not None:
        if generations_to_plot is None:
            generations_to_plot = len(avg_fitness)
    else:
        print("No data provided to plot.")
        return

    gens = range(1, generations_to_plot + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(gens, avg_fitness[:generations_to_plot], label='Average Fitness', marker='o', linestyle='-')
    plt.plot(gens, best_fitness[:generations_to_plot], label='Best Fitness', marker='x', linestyle='--')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Fitness Over Generations (Run ID: {run_id})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    results_dir = f"results_{run_id}"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f"fitness_plot_{run_id}.png"))
    print(f"Fitness plot saved to {os.path.join(results_dir, f'fitness_plot_{run_id}.png')}")
    plt.show()


def plot_robot_pose_data(run_id, filename="robot_pose_data.csv", ellipse_step=10):
    """
    Plots the robot's true pose, filtered pose (if available), uncertainty ellipses,
    and the maze from the saved CSV and JSON files.

    :param run_id: The ID of the simulation run.
    :type run_id: str.
    :param filename:  The name of the CSV file containing the pose data.
    :type filename: str.
    :param ellipse_step:  Step value to plot uncertainty ellipses periodically.
    :type ellipse_step: int.
    """
    pose_filename = f"results/{run_id}_{filename}"
    maze_filename = pose_filename.replace(".csv", "_maze.json")

    try:
        # Read CSV file into a list of dictionaries
        pose_data = []
        with open(pose_filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pose_data.append({k: float(v) if v != 'None' else None for k, v in row.items()})
    except FileNotFoundError:
        print(f"Error: Pose data file '{pose_filename}' not found.")
        return

    try:
        with open(maze_filename, 'r') as f:
            maze_data = json.load(f)
            maze_grid = np.array(maze_data)
    except FileNotFoundError:
        print(f"Error: Maze data file '{maze_filename}' not found.")
        maze_grid = None  # Continue plotting poses even if maze is missing

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    if maze_grid is not None:
        height, width = maze_grid.shape
        # Plot obstacles as black squares
        for r in range(height):
            for c in range(width):
                if maze_grid[r, c] == 1:  # Assuming 1 is an obstacle
                    # Draw rectangle at (c * Config.CELL_SIZE, r * Config.CELL_SIZE) with size Config.CELL_SIZE
                    rect = plt.Rectangle((c * Config.CELL_SIZE, r * Config.CELL_SIZE), Config.CELL_SIZE, Config.CELL_SIZE, color='black')
                    ax.add_patch(rect)

        # Set axes limits based on Config dimensions
        ax.set_xlim(0, Config.GRID_WIDTH * Config.CELL_SIZE)
        ax.set_ylim(0, Config.GRID_HEIGHT * Config.CELL_SIZE)


    # Extract true pose data
    true_x = [d['true_x'] for d in pose_data]
    true_y = [d['true_y'] for d in pose_data]
    plt.plot(true_x, true_y, label='True Pose', color='blue')


    # Check if filtered pose data exists (not None)
    if any(d['filter_x'] is not None for d in pose_data):
        filter_x = [d['filter_x'] for d in pose_data if d['filter_x'] is not None]
        filter_y = [d['filter_y'] for d in pose_data if d['filter_y'] is not None]
        # Plot filtered pose as a dashed line
        plt.plot(filter_x, filter_y, label='Filtered Pose', color='red', linestyle='-.')


    # Plot uncertainty ellipses if data exists
    if any(d['uncertainty_major'] is not None for d in pose_data):
        # ax = plt.gca()
        # Plot ellipses periodically
        for i in range(0, len(pose_data), ellipse_step):
            d = pose_data[i]
            if d['uncertainty_major'] is not None:
                center = (d['filter_x'], d['filter_y'])
                semi_major = d['uncertainty_major']
                semi_minor = d['uncertainty_minor']
                angle_rad = d['uncertainty_angle']
                angle_deg = np.degrees(angle_rad)

                ellipse = Ellipse(xy=center, width=0.5 * semi_major, height=0.5 * semi_minor,
                                  angle=angle_deg, edgecolor='green', fc='None', lw=1)
                ax.add_patch(ellipse)


    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("Robot Pose and Uncertainty Ellipses")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  
    plt.show()
    fig.savefig(f"results/{run_id}_robot_pose_data.png")


def save_generation_fitness(run_id, generation_num, avg_fitness, best_fitness, filename="training_fitness_log.csv"):
    """
    Saves the average and the best fitness for a given generation to a CSV file.

    :param run_id: The ID of the overall training run.
    :type run_id: str.
    :param generation_num: The current generation number.
    :type generation_num: int.
    :param avg_fitness: The average fitness of the population in this generation.
    :type avg_fitness: float.
    :param best_fitness: The best fitness achieved in this generation.
    :type best_fitness: float.
    :param filename: The name of the CSV file to save fitness data.
    :type filename: str
    """
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    filepath = os.path.join(results_dir, f"{run_id}_{filename}")
    file_exists = os.path.isfile(filepath)

    data_row = {
        "generation": generation_num,
        "average_fitness": avg_fitness,
        "best_fitness": best_fitness
    }

    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_row.keys())
        if not file_exists or os.path.getsize(filepath) == 0:  # Check if the file is new or empty
            writer.writeheader()
        writer.writerow(data_row)

def save_model(run_id, model, filename="robot_brain.pt"):
    """
    Saves the model state dictionary to a file. If the file already exists,
    it appends a timestamp to create a unique filename.

    :param run_id: The ID of the overall training run.
    :type run_id: str
    :param model: The model to save.
    :type model: torch.nn.Module
    :param filename: The base filename to save the model as.
    :type filename: str
    """
    filename = f"results/{run_id}_{filename}.pt"
    torch.save(model.state_dict(), filename)


def load_model(run_id=None, model=None, filename="robot_brain.pt", filepath=None):
    """
    Loads a saved model state dictionary into a PyTorch model.

    :param run_id: The ID of the training run to load.
    :type run_id: str, optional
    :param model: The model instance to load weights into.
    :type model: torch.nn.Module
    :param filename: The base filename of the saved model.
    :type filename: str, optional
    :param filepath: Direct filepath to the model, overriding run_id and filename.
    :type filepath: str, optional
    :returns torch.nn.Module: The model with loaded weights.
    """
    if filepath is None:
        if run_id is None:
            raise ValueError("Either run_id or filepath must be provided")
        filepath = f"results/{run_id}_{filename}"

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")

    if model is None:
        raise ValueError("A model instance must be provided to load weights into")

    # Load state dictionary
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)

    print(f"Model successfully loaded from {filepath}")

def normalize(value, min_val, max_val):
    EPS = 1e-6
    return (value - min_val) / (max_val - min_val + EPS)

def plot_pareto_solutions(
    solutions_to_plot: list, # List of Individual-like objects
    objective_indices: list[int],
    all_objective_names: list[str],
    title_suffix: str = ""
):
    """
    Plots the Pareto front for the given solutions and objective indices.

    :param solutions_to_plot: A list of Individual objects (or objects with an 'objectives' attribute)
                           that are on the Pareto front and have valid objectives for plotting.
    :type solutions_to_plot: list
    :param objective_indices: A list of 2 or 3 indices specifying which objectives to plot.
    :type objective_indices: list[int]
    :param all_objective_names: A list of names for all possible objectives.
    :type all_objective_names: list[str]
    :param title_suffix: Optional. A string to append to the plot title.
    :type title_suffix: Optional(str)
    """
    if not solutions_to_plot:
        print("No solutions provided to plot.")
        return

    if not (2 <= len(objective_indices) <= 3):
        # This check should ideally happen before calling this function,
        # but good to have a safeguard.
        print("Error: objective_indices must contain 2 or 3 integer indices for plotting.")
        return

    # Extract data for the selected objectives
    obj_data = [[sol.objectives[i] for i in objective_indices] for sol in solutions_to_plot]

    plt.figure(figsize=(10, 8))
    plot_title = f"{len(objective_indices)}D Pareto Front"
    if title_suffix:
        plot_title += f" ({title_suffix})"

    # Get names for the currently plotted objectives
    plotted_obj_names = [all_objective_names[i] if i < len(all_objective_names) else f"Objective {i+1}"
                         for i in objective_indices]

    if len(objective_indices) == 2:
        x_vals = [data[0] for data in obj_data]
        y_vals = [data[1] for data in obj_data]
        plt.scatter(x_vals, y_vals, c='blue', marker='o', label='Non-Dominated Solutions')
        plt.xlabel(plotted_obj_names[0])
        plt.ylabel(plotted_obj_names[1])
        plt.title(plot_title)

    elif len(objective_indices) == 3:
        ax = plt.axes(projection='3d')
        x_vals = [data[0] for data in obj_data]
        y_vals = [data[1] for data in obj_data]
        z_vals = [data[2] for data in obj_data]
        ax.scatter3D(x_vals, y_vals, z_vals, c='blue', marker='o', label='Non-Dominated Solutions')
        ax.set_xlabel(plotted_obj_names[0])
        ax.set_ylabel(plotted_obj_names[1])
        ax.set_zlabel(plotted_obj_names[2])
        plt.title(plot_title)

    plt.legend()
    plt.grid(True)
    plt.show()


def save_pareto_history(run_id, non_dominated_sol, generation_num):
    """
    Saves the current Pareto front for historical tracking.

    :param run_id: ID of the current training run
    :type run_id: str
    :param non_dominated_sol: List of non-dominated solutions (individuals)
    :type non_dominated_sol: list
    :param generation_num: Current generation number
    :type generation_num: int
    """
    # Prepare directory
    history_dir = f"results/{run_id}_pareto_history"
    os.makedirs(history_dir, exist_ok=True)

    # Save objectives data for this generation
    data = []
    for ind in non_dominated_sol:
        data.append({
            "generation": generation_num,
            "objectives": ind.objectives
        })

    # Save as JSON
    with open(f"{history_dir}/gen_{generation_num}.json", 'w') as f:
        json.dump(data, f, indent=2)

    # Also save latest combined file with all generations
    if os.path.exists(f"{history_dir}/combined.json"):
        with open(f"{history_dir}/combined.json", 'r') as f:
            all_data = json.load(f)
    else:
        all_data = []

    all_data.extend(data)
    with open(f"{history_dir}/combined.json", 'w') as f:
        json.dump(all_data, f, indent=2)


def plot_pareto_evolution(run_id, objective_indices, objective_names=None, interval=5, results_dir="results"):
    """
    Creates a 3D plot showing the evolution of the Pareto front across generations.

    :param run_id: ID of the training run
    :param objective_indices: Indices of objectives to plot (list of 3 indices)
    :param objective_names: Names of the objectives for labels
    :param interval: Plot every Nth generation
    :param results_dir: Directory containing results
    """
    if len(objective_indices) != 3:
        print("Error: Need exactly 3 objective indices for 3D Pareto evolution plot")
        return

    pareto_history_dir = f"{results_dir}/{run_id}_pareto_history"
    if not os.path.exists(f"{pareto_history_dir}/combined.json"):
        print(f"Error: No Pareto history found for run {run_id}")
        return

    # Load all generations data
    with open(f"{pareto_history_dir}/combined.json", 'r') as f:
        all_data = json.load(f)

    # Group by generation
    generations = {}
    for entry in all_data:
        gen = entry["generation"]
        if gen % interval == 0:  # Only plot every Nth generation
            if gen not in generations:
                generations[gen] = []
            generations[gen].append(entry["objectives"])

    if not generations:
        print("No data to plot after filtering by interval")
        return

    # Set up the plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create colormap for generations
    gen_nums = sorted(generations.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(gen_nums)))

    # Plot each generation with a different color
    for i, gen in enumerate(gen_nums):
        gen_data = generations[gen]
        if not gen_data:
            continue

        # Extract data for the selected objectives
        x_vals = [data[objective_indices[0]] for data in gen_data]
        y_vals = [data[objective_indices[1]] for data in gen_data]
        z_vals = [data[objective_indices[2]] for data in gen_data]

        # Plot this generation
        ax.scatter(x_vals, y_vals, z_vals, color=colors[i],
                   label=f"Gen {gen}", alpha=0.7, edgecolors='w', s=50)

    # Labels and title
    if objective_names and len(objective_names) > max(objective_indices):
        ax.set_xlabel(objective_names[objective_indices[0]])
        ax.set_ylabel(objective_names[objective_indices[1]])
        ax.set_zlabel(objective_names[objective_indices[2]])
    else:
        ax.set_xlabel(f"Objective {objective_indices[0] + 1}")
        ax.set_ylabel(f"Objective {objective_indices[1] + 1}")
        ax.set_zlabel(f"Objective {objective_indices[2] + 1}")

    ax.set_title(f"Evolution of Pareto Front Across Generations (Run {run_id})")

    # Add legend with generation numbers
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))

    # Save figure
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{run_id}_pareto_evolution.png", dpi=300)
    print(f"Saved Pareto evolution plot to {results_dir}/{run_id}_pareto_evolution.png")
    plt.show()

plot_fitness_progress()