import numpy as np


class LinearObj:
    def __init__(self, vx: float, vy: float, vz: float, noise: float):
        self.r = np.array([0., 0., 0., vx, vy, vz])
        self.noise = noise

    def move(self, dt: float):
        self.r[0:3] += self.r[3:] * dt

    def sensor_data(self):
        return self.r[0:3] + np.random.normal(0, self.noise, 3)

    def track_data(self):
        return np.copy(self.r[0:3])


class CircularObj:
    def __init__(self, r: float, f: float, noise: float):
        self.r = np.array([r, 0., 0.])
        self.rmag = r
        self.noise = noise
        self.w = 2 * np.pi * f
        self.time = 0

    def move(self, dt: float):
        self.time += dt
        self.r[0] = self.rmag*np.cos(self.w * self.time)
        self.r[1] = self.rmag*np.sin(self.w * self.time)

    def sensor_data(self):
        return self.r[0:3] + np.random.normal(0, self.noise, 3)

    def track_data(self):
        return np.copy(self.r[0:3])

    def reset(self):
        self.r = np.array([self.rmag, 0., 0.])
        self.time = 0

