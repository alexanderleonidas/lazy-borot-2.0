import pygame
import time
from maps import Maps
from picasso import Picasso
from config import Config
from robot import Robot, Action
from utils import save_run, plot_robot_pose_data, load_model
from controller import RobotBrain, device

def main(save_results=False, plot_results=False):
    run_id = str(int(time.time()))
    fps = 30
    pygame.init()
    screen = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
    picasso = Picasso(screen)
    pygame.display.set_caption("Robot Simulator")
    # Create a robot instance at the starting position.
    robot = Robot(Config.start_pos[0], Config.start_pos[1], 0, filter_type='EKF', mapping=False, ann=False)
    if hasattr(robot, 'ann'):
        brain = RobotBrain().to(device)
        load_model(run_id='1746749158', model=brain)
    # Main simulation loop
    running = True
    time_step = 0
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYUP and not hasattr(robot, 'ann'):
                # --- Keyboard Controls ---
                # W/S keys control left wheel forward/backward velocity.
                # O/K keys control right wheel forward/backward velocity.
                # Space bar stops the robot.w
                if event.key == pygame.K_o:
                    robot.set_velocity(Action.INCREASE_RIGHT)
                if event.key == pygame.K_k:
                    robot.set_velocity(Action.DECREASE_RIGHT)
                if event.key == pygame.K_w:
                    robot.set_velocity(Action.INCREASE_LEFT)
                if event.key == pygame.K_s:
                    robot.set_velocity(Action.DECREASE_LEFT)
                if event.key == pygame.K_SPACE:
                    robot.set_velocity(Action.BREAK)
            else:
                # --- ANN Control ---.
                if hasattr(robot, 'ann'):
                    ann_inputs = robot.get_ann_inputs()
                    action = brain.predict(ann_inputs)
                    robot.set_velocity(action)

        # Update the robot's state with a fixed time step.
        dt = 1/fps
        time_step += 1
        robot.update_motion(dt, Config.maze_grid)
        if hasattr(robot, 'filter'):
            robot.filter.pose_tracking(dt)
        if hasattr(robot, 'mapping'):
            robot.mapping.update(robot.filter.pose, robot.sensor_readings)
        if save_results:
            save_run(run_id, robot, time_step, filter_instance=robot.filter)
        # --- Rendering ---
        picasso.draw_map(robot, show_sensors=True, show_dust=True)
        picasso.update_display(fps)
        # Maps.add_noise_to_maze(Config.maze_grid, dt*0.1) # Make obstacles move in the maze with this line
    if save_results: save_run(run_id, robot, time_step+1, filter_instance=robot.filter, maze=Config.maze_grid)
    if plot_results: plot_robot_pose_data(run_id)
    picasso.quit()

if __name__ == "__main__":
    main(False, False)