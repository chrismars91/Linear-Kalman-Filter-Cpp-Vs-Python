import numpy as np
from kalmanfilter import KalmanFitler
from things_that_move import CircularObj
import kalmanfilter_cpp
from timeit import default_timer as timer

noise = 5.0
time = np.linspace(0, 200, 20001)
obj1 = CircularObj(200.0, .005, noise=noise)
dt = time[1] - time[0]

kf = KalmanFitler(sensor_noise_std=noise, processes_model_std=0.1)
kfcpp = kalmanfilter_cpp.KF(noise, 0.1)

start = timer()
for t in time:
    obj1.move(dt)
    kfcpp.predict(dt)
    kfcpp.update(np.array([obj1.sensor_data()]).T)
end = timer()
print(f"c++: {end - start}")


obj1.reset()

start = timer()
for t in time:
    obj1.move(dt)
    kf.predict(dt)
    kf.update(np.array([obj1.sensor_data()]).T)
end = timer()
print(f"numpy w/ python: {end - start}")


"""
c++: 0.06791720900037035
numpy w/ python: 0.3809910410000157
"""


