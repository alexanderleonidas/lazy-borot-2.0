import numpy as np
from math import cos, sin, atan2, sqrt, acos, pi
from itertools import combinations
import pygame


class KalmanFilter:
    def __init__(self, robot):
        self.robot = robot
        self.pose = self.robot.get_pose()
        self.covariance = np.eye(3) * 0.01
        self.A = np.eye(3)  # State transition matrix
        self.B = np.zeros((3, 2))  # Control input matrix
        
        self.R = np.eye(3) * 0.1 # Process noise covariance
        self.C = np.eye(3)
        self.Q = np.eye(3) * 0.1 # Measurement noise covariance
        self.I = np.eye(3)

        self.belief_history = []
        self.uncertainty_history = []
        self.max_history_length = 200

    def pose_tracking(self, dt):
        """
        Performs the Kalman filter prediction and correction steps.
        """

        # Prediction step
        u = np.array([self.robot.get_speed(),
                      self.robot.get_angular_velocity()])
        
        theta = self.pose[2]

        self.B = np.array([
            [dt * cos(theta), 0],
            [dt * sin(theta), 0],
            [0, dt]
        ])

        self.pose = np.dot(self.A, self.pose) + np.dot(self.B, u)
        self.pose[2] = self.pose[2] % (2 * pi)
        self.covariance = np.matmul((np.matmul(self.A, self.covariance)), self.A.T) + self.R

        # Correction step

        # Correct the predicted pose through the visible landmarks
        triangulated_pose = self.triangulate_pose_from_landmarks()
        if triangulated_pose is None:
            # print("\rNo visible landmarks — correction skipped.", end='', flush=True)
            self.belief_history.append(self.pose)
            self.calculate_uncertainty_ellipse()
            if len(self.belief_history) > self.max_history_length:
                self.belief_history.pop(0)
            return
        # print("\rVisible landmarks detected — correction applied.", end='', flush=True)
        else:
            # Kalman gain
            k_gain = np.matmul(np.matmul(self.covariance, self.C.T),
                          np.linalg.inv(np.matmul(np.matmul(self.C, self.covariance), self.C.T) + self.Q))

            self.pose = self.pose + np.matmul(k_gain, (triangulated_pose - np.dot(self.C, self.pose)))
            self.pose[2] = self.pose[2] % (2 * pi)
            self.covariance = np.matmul((self.I - np.matmul(k_gain, self.C)), self.covariance)

            self.belief_history.append(self.pose)
            self.calculate_uncertainty_ellipse()
            # if len(self.belief_history) > self.max_history_length:
            #     self.belief_history.pop(0)
            # if len(self.uncertainty_history) > 20:
            #     self.uncertainty_history.pop(0)

    def triangulate_pose_from_landmarks(self):
        """
        Performs triangulation using the FIRST valid pair of landmarks found.
        Returns:
            numpy.ndarray: Estimated pose [x, y, theta] with noise, or None if insufficient landmarks or no valid pair found.
        WARNING: Using only one pair can be sensitive to measurement noise.
        """
        if len(self.robot.visible_measurements) < 2:
            # print("\rTriangulation requires at least 2 landmarks.", end='', flush=True)
            return None  # Need at least two landmarks to triangulate

        calculated_pose = None  # Variable to store the result from the first valid pair

        # Iterate through all unique pairs of visible landmarks
        for (z1, lm1), (z2, lm2) in combinations(self.robot.visible_measurements, 2):
            try:
                d1, b1 = z1
                d2, b2 = z2
                x1, y1 = lm1
                x2, y2 = lm2

                # Basic validation: distances should be positive
                if d1 <= 0 or d2 <= 0:
                    continue

                # --- Geometric Triangulation Logic ---
                landmark_dist_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
                landmark_dist = sqrt(landmark_dist_sq)

                if landmark_dist < 1e-6:  # Avoid division by zero / numerical instability
                    continue

                # Use law of cosines
                cos_alpha1 = (d1 ** 2 + landmark_dist_sq - d2 ** 2) / (2 * d1 * landmark_dist)
                cos_alpha1 = np.clip(cos_alpha1, -1.0, 1.0)  # Clamp for numerical stability
                alpha1 = acos(cos_alpha1)

                angle_lm1_lm2 = atan2(y2 - y1, x2 - x1)

                # Calculate two possible robot positions
                angle1 = angle_lm1_lm2 - alpha1
                rx_candidate1 = x1 + d1 * cos(angle1)
                ry_candidate1 = y1 + d1 * sin(angle1)

                angle2 = angle_lm1_lm2 + alpha1
                rx_candidate2 = x1 + d1 * cos(angle2)
                ry_candidate2 = y1 + d1 * sin(angle2)

                # --- Disambiguation using bearings ---
                bearing1_cand1 = atan2(y1 - ry_candidate1, x1 - rx_candidate1)
                bearing2_cand1 = atan2(y2 - ry_candidate1, x2 - rx_candidate1)
                bearing1_cand2 = atan2(y1 - ry_candidate2, x1 - rx_candidate2)
                bearing2_cand2 = atan2(y2 - ry_candidate2, x2 - rx_candidate2)

                # Wrap angle differences correctly
                err1_1 = (bearing1_cand1 - b1 + np.pi) % (2 * np.pi) - np.pi
                err2_1 = (bearing2_cand1 - b2 + np.pi) % (2 * np.pi) - np.pi
                consistency_err1 = abs((err1_1 - err2_1 + np.pi) % (2 * np.pi) - np.pi)

                err1_2 = (bearing1_cand2 - b1 + np.pi) % (2 * np.pi) - np.pi
                err2_2 = (bearing2_cand2 - b2 + np.pi) % (2 * np.pi) - np.pi
                consistency_err2 = abs((err1_2 - err2_2 + np.pi) % (2 * np.pi) - np.pi)

                # rx, ry = (rx_candidate1+rx_candidate2)/2, (ry_candidate1 + ry_candidate2)/2
                # theta = atan2(sin((err1_1+err1_2)/2) + sin((err2_1+err2_2)/2), np.cos((err1_1+err1_2)/2) + np.cos((err2_1+err2_2)/2))
                # Choose the candidate position and estimate theta
                if consistency_err1 < consistency_err2:
                    rx, ry = rx_candidate1, ry_candidate1
                    # Average theta estimates from this pair
                    theta = atan2(np.sin(err1_1) + np.sin(err2_1), np.cos(err1_1) + np.cos(err2_1))

                else:
                    rx, ry = rx_candidate2, ry_candidate2
                    # Average theta estimates from this pair
                    theta = atan2(np.sin(err1_2) + np.sin(err2_2), np.cos(err1_2) + np.cos(err2_2))

                # --- Store result and break loop ---
                # We found a valid pair and calculated the pose
                calculated_pose = np.array([rx, ry, theta])
                break  # Exit the loop after processing the first valid pair

            except ValueError as e:
                # print(f"\rMath error during triangulation for a pair: {e}", end='', flush=True)
                continue  # Try the next pair
            except Exception as e:
                # print(f"\rUnexpected error during triangulation for a pair: {e}", end='', flush=True)
                continue  # Try the next pair

        # Check if a pose was calculated
        if calculated_pose is None:
            # print("\rNo valid pose could be calculated from any landmark pair.", end='', flush=True)
            return None

        # Add noise to the single calculated pose
        noisy_pose = np.array([
            calculated_pose[0] + np.random.normal(0, 0.1),
            calculated_pose[1] + np.random.normal(0, 0.1),
            calculated_pose[2] + np.random.normal(0, 0.1)
        ])

        return noisy_pose

    def calculate_uncertainty_ellipse(self, confidence_sigma=2.0):
        """
        Calculates the parameters for the uncertainty ellipse based on the
        positional (x, y) covariance.

        Args:
            confidence_sigma (float): The number of standard deviations
                                      to define the ellipse boundary (e.g., 2.0 for ~95%).

        Returns:
            tuple: (semi_major_axis, semi_minor_axis, angle_degrees) or None if invalid.
                   Angle is degrees counter-clockwise from the positive x-axis.
        """
        # Extract the 2x2 covariance matrix for x and y
        cov_xy = self.covariance[0:2, 0:2]

        try:
            # Check if covariance is finite and not all zero
            if not np.all(np.isfinite(cov_xy)) or np.allclose(cov_xy, 0):
                return None

            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(cov_xy)

            # Eigenvalues represent variance along principal axes.
            # Ensure they are non-negative (handle potential float inaccuracies)
            eigenvalues = np.maximum(eigenvalues, 0)

            # Get the index of the largest eigenvalue
            major_idx = np.argmax(eigenvalues)
            minor_idx = 1 - major_idx  # The other index

            # Semi-axis lengths are proportional to sqrt(eigenvalue)
            # Use confidence_sigma to scale (e.g., 2-sigma ellipse)
            semi_major = confidence_sigma * sqrt(eigenvalues[major_idx])
            semi_minor = confidence_sigma * sqrt(eigenvalues[minor_idx])

            # Angle of the major axis is the angle of the corresponding eigenvector
            major_eigenvector = eigenvectors[:, major_idx]
            angle_rad = atan2(major_eigenvector[1], major_eigenvector[0]) % 2*pi

            self.uncertainty_history.append({
                'center': (self.pose[0], self.pose[1]),
                'semi_major': semi_major,
                'semi_minor': semi_minor,
                'angle_rad': angle_rad,
                'timestamp': pygame.time.get_ticks()
            })

            return semi_major, semi_minor, angle_rad

        except np.linalg.LinAlgError:
            # Matrix might be singular or other linear algebra issues
            # print("\rWarning: Could not compute eigenvalues for covariance ellipse.", end='', flush=True)
            return None
        except Exception as e:
            # Catch other potential errors
            # print(f"\rError calculating ellipse parameters: {e}", end='', flush=True)
            return None



    

