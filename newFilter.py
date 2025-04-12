import numpy as np
from math import cos, sin
from config import Config
from robot import Robot
from itertools import combinations

class KalmanFilter:
    def __init__(self, robot):
        self.robot = robot
        self.state = robot.get_pose() # Initial po
        self.Q = np.diag([0.1, 0.1, 0.05]) # Uncertainity about robot's motion
        self.P = np.diag([0.1,0.1,0.1])  # Very low uncertainty because we know where the robot is located initially - covariance matrix
        self.R = np.diag([1.0, 1.0, 0.1])
        self.belief_history = []
        self.max_history_length = 200  


    def prediction(self):
        dt = 1/30
        v = self.robot.get_linear_velocity()
        omega = self.robot.get_angular_velocity()
        theta = self.state[2]

        dx = dt * v * cos(theta)
        dy = dt * v * sin(theta)
        dtheta = dt * omega

        x_new = self.state[0] + dx
        y_new = self.state[1] + dy
        theta_new = theta + dtheta

        # Update the state
        print(f"[PREDICT] State before: {self.state}")
        self.state = np.array([x_new, y_new, theta_new])
        print(f"[PREDICT] State after prediction: {[x_new, y_new, theta_new]}")
        

        # Update the uncertainty (assuming linear model F = I)
        self.P = self.P + self.Q

    def correction(self, visible_landmarks):
        triangulated_pose = self.triangulate_pose_from_landmarks(visible_landmarks)
        if triangulated_pose is None:
            print("No visible landmarks — correction skipped.\n")
            return

        # Innovation (measurement residual)
        z = triangulated_pose
        h = self.state
        y_residual = z - h
        y_residual[2] = self.normalize_angle(y_residual[2])

        # Measurement model H (identity)
        H = np.eye(3)

        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ y_residual
        self.state[2] = self.normalize_angle(self.state[2])
        
        # Update covariance
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P


    def triangulate_pose_from_landmarks(self, visible_landmarks):
        """
        Performs triangulation: in the case of more than two landmarks it takes the average of the estimates,
        including orientation (theta) from bearings.
        """
        if len(visible_landmarks) < 2:
            return None  # Need at least two landmarks to triangulate

        pos_estimates = []
        theta_estimates = []

        for (z1, lm1), (z2, lm2) in combinations(visible_landmarks, 2):
            d1, b1 = z1
            d2, b2 = z2
            x1, y1 = lm1
            x2, y2 = lm2

            # Estimate robot position relative to each landmark
            rx1 = x1 - d1 * np.cos(b1)
            ry1 = y1 - d1 * np.sin(b1)
            rx2 = x2 - d2 * np.cos(b2)
            ry2 = y2 - d2 * np.sin(b2)

            rx = (rx1 + rx2) / 2
            ry = (ry1 + ry2) / 2
            pos_estimates.append([rx, ry])

            # Estimate orientation from each bearing
            theta1 = np.arctan2(y1 - ry, x1 - rx) - b1
            theta2 = np.arctan2(y2 - ry, x2 - rx) - b2

            theta1 = self.normalize_angle(theta1)
            theta2 = self.normalize_angle(theta2)
            theta_estimates.append(theta1)
            theta_estimates.append(theta2)

        mean_pos = np.mean(pos_estimates, axis=0)
        mean_theta = np.mean(theta_estimates)
        mean_theta = self.normalize_angle(mean_theta)

        pos_noise = np.random.normal(0, [0.5, 0.5])
        theta_noise = np.random.normal(0, 0.05)
        noisy_theta = self.normalize_angle(mean_theta + theta_noise)

        noisy_pose = np.array([
            mean_pos[0] + pos_noise[0],
            mean_pos[1] + pos_noise[1],
            noisy_theta
        ])

        # Store the belief history
        self.belief_history.append(noisy_pose)
        if len(self.belief_history) > self.max_history_length:
            self.belief_history.pop(0)
        # Return the noisy pose

        return noisy_pose

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
