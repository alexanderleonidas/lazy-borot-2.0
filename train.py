import time
import pygame
import copy
from config import Config
from maps import Maps
from picasso import Picasso 
from robot import Robot
from controller import Evolution 
from utils import save_generation_fitness, save_model, plot_fitness_progress, save_pareto_history, plot_pareto_evolution
import os



def train(save_results=False, plot_results=False, show_screen=False, multi_objective=False):
    run_id = str(int(time.time()))
    fps = 30 # Simulation speed, not necessarily display FPS if unthrottled

    # Evolutionary Parameters
    population_size = 40
    selection_percentage = 0.5
    error_range = 0.0001
    mutate_percentage = 0.15
    time_steps = 3000  # Duration of each individual's simulation
    generations = 100

    # Initialize Evolution
    evolution = Evolution(population_size, selection_percentage, error_range, mutate_percentage, multi_objective=multi_objective, rnn=True)

    avg_fitness_over_generations = []
    best_fitness_over_generations = []

    for generation in range(generations):
        print(f"--- Generation {generation + 1}/{generations} ---")

        generation_fitness_values = []

        original_dust_particles = copy.deepcopy(Config.dust_particles)

        for i, individual in enumerate(evolution.population):
            if show_screen:
                pygame.init()
                screen = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
                picasso = Picasso(screen)  # Assuming Picasso can be initialized once
                pygame.display.set_caption(f"Gen {generation + 1}, Indiv {i+1}/{population_size}")

            # Initialize robot for the current individual
            # Ensure Robot class correctly uses individual.brain as its controller
            # Config.dust_particles = Maps.generate_static_dust(Config.GRID_WIDTH, Config.GRID_HEIGHT, Config.maze_grid,
            #                                                   Config.CELL_SIZE, Config.NEON_PINK)
            start_pos = Maps.find_empty_spot(Config.maze_grid, Config.CELL_SIZE)
            robot = Robot(start_pos[0], start_pos[1], 0,
                         filter_type='EKF', mapping=True, ann=True)  # Use ann=True
            # if hasattr(robot, 'brain'):
            #     robot.brain = individual.brain

            robot.reset_episode_stats()

            dt = 1 / fps # Timestep duration

            for step in range(time_steps):
                if show_screen:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            picasso.quit() # Make sure Picasso handles Pygame quit
                            print("Training interrupted by user.")
                            if plot_results and avg_fitness_over_generations:
                                plot_fitness_progress(avg_fitness_over_generations, best_fitness_over_generations, run_id, generation)
                            return # Exit train function


                # --- Robot Simulation Step ---
                robot.update_motion(dt, Config.maze_grid) # Update kinematics, check collisions

                ann_inputs = robot.get_ann_inputs() # Get sensory data for ANN
                action = individual.brain.continuous_predict(ann_inputs) # ANN decides action
                # print(f"ANN Selected action: {action}")
                robot.set_velocity(action) # Apply action to robot motors

                if show_screen:
                    picasso.draw_map(robot, show_sensors=True, show_dust=True) # Update visualization
                    picasso.update_display(fps) # Control display rate if needed

            # --- End of simulation for one individual ---
            # Fitness should be computed based on the robot's performance over all time_steps
            if multi_objective:
                evolution.compute_individual_objectives(individual, robot)
            else:
                evolution.compute_individual_fitness(individual, robot, weights=[]) # Ensure this uses the robot's final state
            generation_fitness_values.append(individual.fitness())
            print(f"  Individual {i+1} Fitness: {individual.fitness():.4f}")
            Config.dust_particles = copy.deepcopy(original_dust_particles)
            robot.dust_particles = Config.dust_particles

        # --- End of generation ---
        if not generation_fitness_values: # Should not happen if population_size > 0
            print("Warning: No fitness values recorded for this generation.")
            current_avg_fitness = 0
            current_best_fitness = 0
        else:
            current_avg_fitness = sum(generation_fitness_values) / len(generation_fitness_values)
            current_best_fitness = max(generation_fitness_values)

        avg_fitness_over_generations.append(current_avg_fitness)
        best_fitness_over_generations.append(current_best_fitness)

        if (generation + 1) % 3 == 0 and save_results:
            checkpoint_dir = f"results/{run_id}_checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)

            best = max(evolution.population, key=lambda ind: ind.fitness())
            fname = f"checkpoints/{run_id}_gen{generation + 1}.pt"
            save_model(run_id, best.brain, filename=fname)

            print(f"Saved checkpoint: {checkpoint_dir}/{run_id}_gen{generation + 1}.pt")

        # Create the next generation based on fitness
        if multi_objective:
            # evolution.plot_pareto_front([0,1])
            save_pareto_history(run_id, evolution.get_non_dominated_solutions(), generation+1)
            evolution.create_next_generation_mo()
        else:
            evolution.create_next_generation()

        if save_results:
            save_generation_fitness(run_id, generation + 1, current_avg_fitness, current_best_fitness)

        print(f"Generation {generation + 1} completed. Avg Fitness: {current_avg_fitness:.4f}, Best Fitness: {current_best_fitness:.4f}\n")

    # --- End of Training ---
    if show_screen:
        if picasso: picasso.quit()
        pygame.quit()

    print("="*20 + " Training Finished " + "="*20)
    if avg_fitness_over_generations: # Check if any generations were run
        final_avg_fitness = sum(avg_fitness_over_generations) / len(avg_fitness_over_generations)
        final_best_fitness = max(best_fitness_over_generations)
        print(f"Overall Average Fitness: {final_avg_fitness:.4f}")
        print(f"Overall Best Fitness Achieved: {final_best_fitness:.4f}")

        if save_results:
            best_individual_final_pop = max(evolution.population, key=lambda ind: ind.fitness())
            save_model(run_id, best_individual_final_pop.brain)
            print(f"Best model from final population saved for run ID: {run_id}")

        if plot_results:
            plot_fitness_progress(avg_fitness_over_generations, best_fitness_over_generations, run_id)
            plot_pareto_evolution(run_id, [0,1,2])
    else:
        print("No generations were completed.")


if __name__ == "__main__":

    train(save_results=True, plot_results=True, show_screen=False, multi_objective=True) # Set show_screen=True to watch