import numpy as np
import math
from math import cos, sin
from itertools import combinations

class KalmanFilter:
    def __init__(self, robot):
        self.robot = robot
        self.state = robot.get_pose() # x, y, theta - initial state
        self.A = np.eye(3)  # State transition matrix (identity for simplicity)
        self.B = np.zeros((3,2))
        self.C = np.eye(3)
        self.Q = np.eye(3) * 0.001
        self.R = np.eye(3) * 0.001  
        self.covariance = np.eye(3) * 0.001  
        self.belief_history = []
        self.max_history_length = 200  
        self.predicted_state = self.state


    def prediction(self):
        self.predicted_state = self.robot.get_pose()
        dt = 1/30
        v = self.robot.get_linear_velocity()
        omega = self.robot.get_angular_velocity()
        u = np.array([v, omega])

        theta = self.state[2]

        #dx = dt * v * cos(theta)
        #dy = dt * v * sin(theta)
        #dtheta = dt * omega
        
        dx = dt * cos(theta)
        dy = dt * sin(theta)
        dtheta = dt * omega
        
        self.B = np.array([
            [dx, 0],
            [dy, 0],
            [0, dtheta]
        ])

        self.predicted_state = np.dot(self.A, self.predicted_state) + np.dot(self.B, u)

        #x_new = self.state[0] + dx
        #y_new = self.state[1] + dy
        #theta_new = theta + dtheta

        self.covariance = np.matmul((np.matmul(self.A, self.covariance)), self.A.T) + self.R

    def correction(self, visible_landmarks):
        triangulated_pose = self.triangulate_pose_from_landmarks(visible_landmarks)
        if triangulated_pose is None:
            print("\rNo visible landmarks — correction skipped.\n", end='', flush=True)
            return
        z = triangulated_pose

        first_term = np.matmul(self.covariance, self.C.T)
        second_term = np.linalg.inv((np.matmul(np.matmul(self.C, self.covariance), self.C.T) + self.Q))
        K = np.matmul(first_term, second_term)

        self.state = self.predicted_state + np.matmul(K, (z - np.dot(self.C, self.predicted_state)))
        self.covariance = np.matmul((np.eye(3) - np.matmul(K, self.C)), self.covariance)

        # Store the belief history
        self.belief_history.append(self.state)
        if len(self.belief_history) > self.max_history_length:
            self.belief_history.pop(0)



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

    def get_uncertainty_ellipse(self):
        """
        Draws an uncertainty ellipse representing the covariance of the robot's (x, y) position.
        """

        # Extract the 2x2 covariance for position (ignore theta)
        pos_cov = self.covariance[0:2, 0:2]
        
        # Get ellipse parameters from covariance matrix
        eigenvals, eigenvecs = np.linalg.eigh(pos_cov)
        order = eigenvals.argsort()[::-1]
        eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

        angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
        angle_deg = np.degrees(angle)

        # Scale eigenvalues to draw an ellipse with a 95% confidence interval (~2 std dev)
        chisquare_scale = 5.991  # 95% confidence interval for 2 DoF
        width, height = 2 * np.sqrt(eigenvals * chisquare_scale)

        center_x = int(self.state[0])
        center_y = int(self.state[1])

        return center_x, center_y, width, height, angle_deg
    
    def ellipse(self):

        """
        Computes the uncertainty ellipse parameters (x, y, theta_deg) from the 2x2 position covariance matrix.
        
        Returns:
            x (float): semi-major axis length
            y (float): semi-minor axis length
            theta_deg (float): orientation angle in degrees
        """
        # Extract positional covariance entries
        a = self.covariance[0][0]  # Var(x)
        b = self.covariance[0][1]  # Cov(x, y)
        c = self.covariance[1][1]  # Var(y)

        # Analytical eigenvalue calculation for symmetric 2x2 matrix
        l1 = (a + c) / 2 + np.sqrt(((a - c) / 2) ** 2 + b ** 2)
        l2 = (a + c) / 2 - np.sqrt(((a - c) / 2) ** 2 + b ** 2)

        # Orientation of the ellipse
        if b == 0 and a >= c:
            theta = 0
        elif b == 0 and a < c:
            theta = np.pi / 2
        else:
            theta = math.atan2(l1 - a, b)

        # Axes lengths (standard deviation scale)
        x = np.sqrt(abs(l1))
        y = np.sqrt(abs(l2))
        theta_deg = math.degrees(theta)

        return (x, y, theta_deg)