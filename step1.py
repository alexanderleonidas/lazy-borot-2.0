import pygame
from picasso import Picasso
from config import Config
from robot import Robot, Action

def main():
    fps = 30
    pygame.init()
    screen = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
    picasso = Picasso(screen)
    pygame.display.set_caption("Robot Simulator (step 1)")
    # Create a robot instance at the starting position.
    robot = Robot(Config.start_pos[0], Config.start_pos[1], 0)

    # Main simulation loop
    running = True
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYUP:
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

        # Update the robot's state with a fixed time step.
        dt = 1/fps
        robot.update_motion(dt, Config.maze_grid)
        robot.filter.pose_tracking(dt)

        # --- Rendering ---
        picasso.draw_map(robot, belief_history=robot.filter.belief_history)
        picasso.update_display(fps)
    picasso.quit()

if __name__ == "__main__":
    main()