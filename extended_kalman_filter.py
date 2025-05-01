import numpy as np
from math import cos, sin, atan2, sqrt, acos, degrees, radians, pi # Added radians, pi

import pygame # Keep for timestamping if needed

class ExtendedKalmanFilter: # Renamed for clarity
    def __init__(self, robot, initial_covariance=0.01, process_noise_std=(1, 1, np.radians(1)), measurement_noise_std=(1, np.radians(5))):
        """
        Initializes the Extended Kalman Filter.

        Args:
            robot: The robot object providing pose, velocity, and measurements.
            initial_covariance (float): Initial uncertainty diagonal value.
            process_noise_std (tuple): Standard deviations for process noise (x, y, theta).
            measurement_noise_std (tuple): Standard deviations for measurement noise (distance, bearing).
        """
        self.robot = robot
        # Initial state estimate (using robot's initial pose)
        self.pose = self.robot.get_pose().astype(float) # Ensure a float type
        # Initial state covariance
        self.covariance = np.eye(3) * initial_covariance

        # Process Noise Covariance (R) - reflects uncertainty in the motion model
        self.R = np.diag([process_noise_std[0]**2,
                          process_noise_std[1]**2,
                          process_noise_std[2]**2])

        # Measurement Noise Covariance (Q) - reflects uncertainty in sensor measurements
        # Assumes measurements are [distance, bearing]
        self.Q = np.diag([measurement_noise_std[0]**2,
                          measurement_noise_std[1]**2])

        self.I = np.eye(3) # Identity matrix

        self.belief_history = []
        self.uncertainty_history = []
        self.max_history_length = 200 # Keep history length manageable

    def _normalize_angle(self, angle):
        """Normalize angle to be within [-pi, pi)."""
        while angle >= pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle

    def pose_tracking(self, dt):
        """
        Performs the EKF prediction and correction steps.
        """
        # --- EKF Prediction Step ---

        v = self.robot.get_linear_velocity()
        omega = self.robot.get_angular_velocity()
        theta = self.pose[2]

        # 1. Predict state using non-linear motion model f(x, u)
        delta_x = v * dt * cos(theta)
        delta_y = v * dt * sin(theta)
        delta_theta = omega * dt

        predicted_pose = self.pose + np.array([delta_x, delta_y, delta_theta])
        predicted_pose[2] = self._normalize_angle(predicted_pose[2]) # Normalize angle

        # 2. Calculate Jacobian of motion model F = ∂f/∂x
        #    Evaluated at the *current* state self.pose (or predicted_pose, conventions vary)
        F = np.array([
            [1.0, 0.0, -v * dt * sin(theta)],
            [0.0, 1.0,  v * dt * cos(theta)],
            [0.0, 0.0, 1.0]
        ])

        # 3. Predict covariance P_k|k-1 = F * P_k-1|k-1 * F^T + R
        predicted_covariance = F @ self.covariance @ F.T + self.R

        # Update internal state *after* using the current state for the Jacobin etc.
        self.pose = predicted_pose
        self.covariance = predicted_covariance


        # --- EKF Correction Step (Iterate through visible landmarks) ---

        num_landmarks_processed = 0
        # Make a copy to avoid issues if the list changes during iteration
        visible_measurements = list(self.robot.visible_measurements)

        if not visible_measurements:
            # print("\rNo visible landmarks — correction skipped.", end='', flush=True)
            pass # Skip correction if no landmarks are seen
        else:
            # print(f"\rProcessing {len(visible_measurements)} landmarks...", end='', flush=True)
            for z_measured, landmark_pos in visible_measurements:
                d_measured, b_measured = z_measured # Actual measurement [distance, bearing]
                lx, ly = landmark_pos              # Landmark absolute position

                # Use the *predicted* state for calculating expected measurement and Jacobian H
                current_x = self.pose[0]
                current_y = self.pose[1]
                current_theta = self.pose[2]

                # 1. Calculate expected measurement h(x_k|k-1)
                delta_lx = lx - current_x
                delta_ly = ly - current_y
                q = delta_lx**2 + delta_ly**2 # Squared distance
                d_expected = sqrt(q)
                # Expected bearing: angle to landmark - robot's angle
                b_expected = self._normalize_angle(atan2(delta_ly, delta_lx) - current_theta)

                z_expected = np.array([d_expected, b_expected])

                # 2. Calculate Jacobian of measurement model H = ∂h/∂x
                #    Evaluated at the predicted state self.pose
                #    Ensure q is not too small to avoid division by zero
                if q < 1e-6:
                    # print(f"\rSkipping landmark {landmark_pos} too close for Jacobian.", end='', flush=True)
                    continue # Skip this landmark if too close

                sqrt_q = sqrt(q)
                H = np.array([
                    [-delta_lx / sqrt_q, -delta_ly / sqrt_q, 0],
                    [ delta_ly / q,      -delta_lx / q,     -1]
                ])

                # 3. Calculate Kalman Gain K
                H_cov = H @ self.covariance # Precompute H * P
                S = H_cov @ H.T + self.Q  # Innovation covariance S = H*P*H^T + Q
                try:
                    S_inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    # print(f"\rWarning: S matrix singular for landmark {landmark_pos}. Skipping update.", end='', flush=True)
                    continue # Skip if matrix is singular

                K = self.covariance @ H.T @ S_inv # Kalman Gain K = P * H^T * S^-1

                # 4. Calculate measurement residual (innovation) y = z_measured - z_expected
                y = z_measured - z_expected
                y[1] = self._normalize_angle(y[1]) # Normalize bearing difference

                # 5. Update state estimate x_k|k = x_k|k-1 + K * y
                correction = K @ y
                self.pose = self.pose + correction
                self.pose[2] = self._normalize_angle(self.pose[2]) # Normalize angle

                # 6. Update covariance estimate P_k|k = (I - K * H) * P_k|k-1
                # Using Joseph form for better numerical stability:
                # P = (I - KH)P(I - KH)^T + KRK^T
                I_KH = self.I - K @ H
                self.covariance = I_KH @ self.covariance @ I_KH.T + K @ self.Q @ K.T
                # Simpler form (potentially less stable):
                # self.covariance = (self.I - K @ H) @ self.covariance

                num_landmarks_processed += 1
                break

        # print(f"\rEKF update complete. Processed {num_landmarks_processed} landmarks. Pose: [{self.pose[0]:.2f}, {self.pose[1]:.2f}, {degrees(self.pose[2]):.1f}]", end='', flush=True)


        # --- Store History ---
        self.belief_history.append(self.pose.copy()) # Store a copy
        self.calculate_uncertainty_ellipse() # Update uncertainty visualization data
        # Limit history size
        if len(self.belief_history) > self.max_history_length:
            self.belief_history.pop(0)
        if len(self.uncertainty_history) > 20: # Also limit uncertainty history
             self.uncertainty_history.pop(0)


    # --- Triangulation function is no longer needed for EKF correction ---
    # def triangulate_pose_from_landmarks(self):
    #     ... REMOVED ...


    def calculate_uncertainty_ellipse(self, confidence_sigma=2.0):
        """
        Calculates the parameters for the uncertainty ellipse based on the
        positional (x, y) covariance. (Unchanged from original)

        Args:
            confidence_sigma (float): The number of standard deviations
                                      to define the ellipse boundary (e.g., 2.0 for ~95%).

        Returns:
            tuple: (semi_major_axis, semi_minor_axis, angle_degrees) or None if invalid.
                   Angle is degrees counter-clockwise from the positive x-axis.
        """
        cov_xy = self.covariance[0:2, 0:2]
        try:
            if not np.all(np.isfinite(cov_xy)) or np.allclose(cov_xy, 0): return None
            eigenvalues, eigenvectors = np.linalg.eig(cov_xy)
            eigenvalues = np.maximum(eigenvalues, 0)
            major_idx = np.argmax(eigenvalues)
            minor_idx = 1 - major_idx
            semi_major = confidence_sigma * sqrt(eigenvalues[major_idx])
            semi_minor = confidence_sigma * sqrt(eigenvalues[minor_idx])
            major_eigenvector = eigenvectors[:, major_idx]
            angle_rad = atan2(major_eigenvector[1], major_eigenvector[0])
            angle_deg = degrees(angle_rad)

            self.uncertainty_history.append({
                'center': (self.pose[0], self.pose[1]),
                'semi_major': semi_major,
                'semi_minor': semi_minor,
                'angle_deg': angle_deg,
                'timestamp': pygame.time.get_ticks() # Use pygame time if available
            })
            # Apply history limit inside the main loop now

            return semi_major, semi_minor, angle_deg
        except np.linalg.LinAlgError: return None
        except Exception as e: return None # Catch other errors


    # Helper to get the current estimated pose
    def get_estimated_pose(self):
        return self.pose.copy()

    # Helper to get current covariance
    def get_covariance(self):
        return self.covariance.copy()