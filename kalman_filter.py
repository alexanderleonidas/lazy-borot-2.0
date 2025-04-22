import numpy as np
import math
from math import cos, sin
from itertools import combinations

class KalmanFilter:
    def __init__(self, robot):
        self.robot = robot
        self.pose = self.robot.get_pose()
        self.covariance = np.eye(3) * 0.001
        self.A = np.eye(3)  # State transition matrix
        self.B = np.zeros((3, 2))  # Control input matrix
        
        self.R = np.eye(3) * 0.001
        self.C = np.eye(3)
        self.Q = np.eye(3) * 0.001
        self.I = np.eye(3)

        self.belief_history = []
        self.max_history_length = 200

    def pose_tracking(self, visible_landmarks):
        """
        Performs the Kalman filter prediction and correction steps.
        """

        # Prediction step
        dt = 1/30
        u = np.array([self.robot.get_linear_velocity(), 
                      self.robot.get_angular_velocity()])
        
        theta = self.pose[2]

        self.B = np.array([
            [dt * cos(theta), 0],
            [dt * sin(theta), 0],
            [0, dt]
        ])

        predicted_pose = np.dot(self.A, self.pose) + np.dot(self.B, u)
        self.covariance = np.matmul((np.matmul(self.A, self.covariance)), self.A.T) + self.R

        # Correction step
        # Kalman gain
        K = np.matmul(np.matmul(self.covariance, self.C.T),
                      np.linalg.inv(np.matmul(np.matmul(self.C, self.covariance), self.C.T) + self.Q))

        # Correct the predicted pose through the visible landmarks
        triangulated_pose = self.triangulate_pose_from_landmarks(visible_landmarks)
        if triangulated_pose is None:
            print("\rNo visible landmarks — correction skipped.\n", end='', flush=True)
            return
        print("\rVisible landmarks detected — correction applied.\n", end='', flush=True)
        
        z = triangulated_pose

        self.pose = predicted_pose + np.matmul(K, (z - np.dot(self.C, predicted_pose)))
        self.covariance = np.matmul((self.I - np.matmul(K, self.C)), self.covariance)

        self.belief_history.append(self.pose)
        if len(self.belief_history) > self.max_history_length:
            self.belief_history.pop(0)

        #return self.pose, self.covariance, self.belief_history

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


            theta_estimates.append(theta1)
            theta_estimates.append(theta2)

        mean_pos = np.mean(pos_estimates, axis=0)
        mean_theta = (np.arctan2(np.sin(theta_estimates).mean(), np.cos(theta_estimates).mean()))

        pos_noise = np.random.normal(0, [0.5, 0.5])
        theta_noise = np.random.normal(0, 0.05)
        noisy_theta = mean_theta + theta_noise

        noisy_pose = np.array([
            mean_pos[0] + pos_noise[0],
            mean_pos[1] + pos_noise[1],
            noisy_theta
        ])

        return noisy_pose
    

