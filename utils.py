import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
import csv
import json
from config import Config


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

                ellipse = Ellipse(xy=center, width=1.5 * semi_major, height=1.5 * semi_minor,
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


