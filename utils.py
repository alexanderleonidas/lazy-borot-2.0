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
def plot_fitness_progress(avg_fitness, best_fitness, run_id, generations_to_plot=None):
    if generations_to_plot is None:
        generations_to_plot = len(avg_fitness)

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

    Args:
        run_id (str): The ID of the simulation run.
        filename (str): The name of the CSV file containing the pose data.
        ellipse_step (int): Step value to plot uncertainty ellipses periodically.
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
    Saves the average and best fitness for a given generation to a CSV file.

    Args:
        run_id (str): The ID of the overall training run.
        generation_num (int): The current generation number.
        avg_fitness (float): The average fitness of the population in this generation.
        best_fitness (float): The best fitness achieved in this generation.
        filename (str): The name of the CSV file to save fitness data.
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
    filename = f"results/{run_id}_{filename}"
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
