import numpy as np
from numpy import dot, zeros, eye
import numpy.linalg as linalg

"""
modeled from https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
"""

class KalmanFitler:
    def __init__(self, sensor_noise_std: float, processes_model_std: float):
        """
        :param sensor_noise_std: what is the expected error in your sensor? this value is easy, how much can
               you trust your sensor.
        :param processes_model_std: what is expected error in your model? This is a linear model, look at the F matrix.
               If the object has any acceleration, the model will need some error to describe it. If the model had
               constant velocity than this KF would be perfect and all your error would be in the sensor or
               sensor_noise_var. Finding the best value can be a trial and error processes.
        """

        self.x = zeros((6, 1))  # state
        self.P = eye(6)  # uncertainty covariance
        self.Q = eye(6)  # process uncertainty
        self.F = eye(6)  # state transition matrix
        self.H = zeros((3, 6))  # measurement function
        self.R = eye(3)  # measurement uncertainty
        self.z = np.array([[None] * 3]).T

        self.K = np.zeros((6, 3))  # kalman gain
        self.y = zeros((3, 1))
        self.S = np.zeros((3, 3))  # system uncertainty
        self.SI = np.zeros((3, 3))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(6)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        self.inv = np.linalg.inv

        self.P *= np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, .1, 0, 0],
            [0, 0, 0, 0, .1, 0],
            [0, 0, 0, 0, 0, .1]
        ])
        self.R *= sensor_noise_std ** 2
        self.Q = np.eye(6) * processes_model_std ** 2
        dt = .001
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

    def predict(self, dt: float):
        self.F[0][3] = dt
        self.F[1][4] = dt
        self.F[2][5] = dt
        F = self.F
        Q = self.Q
        self.x = dot(F, self.x)
        self.P = dot(dot(F, self.P), F.T) + Q
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z: np.ndarray):
        """
        :param z: z.shape == (3, 1)
        """
        R = self.R
        H = self.H
        self.y = z - dot(H, self.x)
        PHT = dot(self.P, H.T)
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        self.K = dot(PHT, self.SI)
        self.x = self.x + dot(self.K, self.y)
        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def get_prediction(self, dt: float):
        F = self.F
        F[0][3] = dt
        F[1][4] = dt
        F[2][5] = dt
        return dot(F, self.x)


"""
update loop logic for KF
/////////////////////////////////////////////////
    kf.predict( dt )
    kf.get_prediction(dt)
    kf.update(z)
"""
