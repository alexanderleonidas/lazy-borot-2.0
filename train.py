import time
import pygame
from config import Config
from picasso import Picasso
from robot import Robot
from controller import Evolution
from utils import save_generation_fitness, save_model
import os

def train(save_results=False, plot_results=False, show_screen=False):
    run_id = str(int(time.time()))
    fps = 30

    population_size = 50
    selection_percentage = 0.5
    error_range = 0.1
    mutate_percentage = 0.2
    time_steps = 500
    generations = 500

    evolution = Evolution(population_size, selection_percentage, error_range, mutate_percentage)

    avg_fitness_over_generations = []
    best_fitness_over_generations = []

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        for i, individual in enumerate(evolution.population):
            if show_screen:
                pygame.init()
                screen = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
                picasso = Picasso(screen)
                pygame.display.set_caption(f"Generation {generation + 1}, Individual {i}")
            robot = Robot(Config.start_pos[0], Config.start_pos[1], 0, filter_type='EKF', mapping=True, ann=True)
            dt = 1 / fps
            for step in range(time_steps):
                if show_screen:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            picasso.quit()
                robot.update_motion(dt, Config.maze_grid)
                if hasattr(robot, 'filter'):
                    robot.filter.pose_tracking(dt)
                if hasattr(robot, 'mapping'):
                    robot.mapping.update(robot.filter.pose, robot.sensor_readings)
                ann_inputs = robot.get_ann_inputs()
                action = individual.brain.predict(ann_inputs)
                robot.set_velocity(action)
                #evolution.compute_individual_fitness_2(individual, robot)
                if show_screen:
                    picasso.draw_map(robot, show_sensors=True, show_dust=True)
                    picasso.update_display(fps)
                # if (step+1) % 100 == 0:
                #     print(f"Individual {i} - Step {step+1}/{time_steps}, Fitness: {individual.fitness:.4f}")
            if show_screen: picasso.quit()
            evolution.compute_individual_fitness_2(individual, robot)

        current_avg_fitness = sum([ind.fitness for ind in evolution.population]) / len(evolution.population)
        current_best_fitness = max([ind.fitness for ind in evolution.population])

        avg_fitness_over_generations.append(current_avg_fitness)
        best_fitness_over_generations.append(current_best_fitness)


        if (generation + 1) % 50 == 0 and save_results:
            checkpoint_dir = f"results/{run_id}_checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)

            best = max(evolution.population, key=lambda ind: ind.fitness)
            fname = f"checkpoints/{run_id}_gen{generation+1}.pth"
            save_model(run_id, best.brain, filename=fname)

            print(f"Saved checkpoint: {checkpoint_dir}/{run_id}_gen{generation+1}.pth")

        evolution.create_next_generation()

        # Save fitness data for the current generation
        if save_results: save_generation_fitness(run_id, generation + 1, current_avg_fitness, current_best_fitness)

        print(f"Generation {generation + 1} completed. Avg Fitness: {current_avg_fitness:.4f}, Best Fitness: {current_best_fitness:.4f}")

    print("Training finished.")
    print(f"Average fitness over generations: {sum(avg_fitness_over_generations) / len(avg_fitness_over_generations)}")
    print(f"Best fitness over generations: {max(best_fitness_over_generations)}")
    if save_results:
        best_individual = max(evolution.population, key=lambda ind: ind.fitness)
        save_model(run_id, best_individual.brain)


if __name__ == "__main__":
    train(save_results=True, show_screen=False)