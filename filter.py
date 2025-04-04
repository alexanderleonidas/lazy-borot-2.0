import numpy as np
import math
from config import Config

class EKF:
    def __init__(self, init_state):
        """
        Initialize the EKF with an initial state vector [x, y, theta] and covariance.
        """
        self.state = init_state  # Estimated state
        self.p = np.eye(3) * 5.0  # Initial covariance matrix

    def predict(self, control, dt):
        """
        Prediction step of the EKF.
        control: [v, omega]
        Uses the robot's motion model to predict the next state.
        """
        v, omega = control
        theta = self.state[2]

        # Motion model: differential drive
        if abs(omega) > 1e-5:
            dx = -(v/omega) * math.sin(theta) + (v/omega) * math.sin(theta + omega * dt)
            dy = (v/omega) * math.cos(theta) - (v/omega) * math.cos(theta + omega * dt)
        else:
            dx = v * math.cos(theta) * dt
            dy = v * math.sin(theta) * dt
        dtheta = omega * dt

        # Predicted state update
        self.state = self.state + np.array([dx, dy, dtheta])
        self.state[2] %= 2 * math.pi

        # Jacobian of the motion model with respect to the state (F)
        f = np.array([[1, 0, -v * math.sin(theta) * dt],
                      [0, 1,  v * math.cos(theta) * dt],
                      [0, 0, 1]])
        # Process noise covariance (Q)
        q = np.diag([0.1, 0.1, 0.05])
        # Covariance update
        self.p = f.dot(self.p).dot(f.T) + q

    def update(self, measurement, maze, sensor_angle):
        """
        Update step of the EKF using a measurement from a sensor.
        Here, we use the front sensor reading (sensor_angle relative to robot's heading).
        measurement: observed distance (scalar)
        maze: the environment map (for ray-casting)
        sensor_angle: relative angle of the sensor (e.g., 0 for the front sensor)
        """
        # Compute expected measurement using ray-casting from the estimated state.
        theta = self.state[2]
        angle = theta + sensor_angle
        expected_distance = 0
        while expected_distance < 100:
            test_x = int((self.state[0] + expected_distance * math.cos(angle)) // Config.CELL_SIZE)
            test_y = int((self.state[1] + expected_distance * math.sin(angle)) // Config.CELL_SIZE)
            if test_x < 0 or test_x >= Config.GRID_WIDTH or test_y < 0 or test_y >= Config.GRID_HEIGHT:
                break
            if maze[test_y, test_x] == 1:
                break
            expected_distance += 1

        # Measurement function h(x) is approximated by expected_distance.
        # For simplicity, we assume h(x) depends primarily on theta.
        # Use numerical differentiation to compute derivative with respect to theta.
        delta = 1e-3
        state_plus = self.state.copy()
        state_plus[2] += delta
        angle_plus = state_plus[2] + sensor_angle
        expected_distance_plus = 0
        while expected_distance_plus < 100:
            test_x = int((state_plus[0] + expected_distance_plus * math.cos(angle_plus)) // Config.CELL_SIZE)
            test_y = int((state_plus[1] + expected_distance_plus * math.sin(angle_plus)) // Config.CELL_SIZE)
            if test_x < 0 or test_x >= Config.GRID_WIDTH or test_y < 0 or test_y >= Config.GRID_HEIGHT:
                break
            if maze[test_y, test_x] == 1:
                break
            expected_distance_plus += 1

        # Derivative of measurement function with respect to theta
        h_theta = (expected_distance_plus - expected_distance) / delta
        # We assume negligible derivatives with respect to x and y.
        h = np.array([[0, 0, h_theta]])
        r = np.array([[1.0]])  # Measurement noise covariance

        s = h.dot(self.p).dot(h.T) + r
        k = self.p.dot(h.T).dot(np.linalg.inv(s))  # Kalman gain

        innovation = measurement - expected_distance
        self.state = self.state + (k.flatten() * innovation)
        self.p = (np.eye(3) - k.dot(h)).dot(self.p)