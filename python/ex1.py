import matplotlib.pyplot as plt
import numpy as np
from kalmanfilter import KalmanFitler
from things_that_move import CircularObj
import kalmanfilter_cpp

"""
create something that will move. It will have circular motion and therefore always be accelerating.
The filter processes_model_std should then not be zero since it is a linear filter!
"""
# thing to track
# /////////////////////////////////////////
noise = 5.0
time = np.linspace(0, 200, 2001)
obj1 = CircularObj(200.0, .005, noise=noise)
dt = time[1] - time[0]  # assume a constant dt

# filters
# /////////////////////////////////////////
processes_model_std = 0.1  # change value to see how the error will change
kf = KalmanFitler(sensor_noise_std=noise, processes_model_std=processes_model_std)
kfcpp = kalmanfilter_cpp.KF(noise, processes_model_std)

# store results
# /////////////////////////////////////////
real_tract = []
track_from_sensor = []
track_from_kf = []

# run test
# /////////////////////////////////////////
for t in time:
    real_tract.append(obj1.track_data())
    track_from_sensor.append(obj1.sensor_data())
    obj1.move(dt)
    kf.predict(dt)
    kfcpp.predict(dt)
    # track_from_kf.append(kf.get_prediction(dt))
    kf.get_prediction(dt)
    track_from_kf.append(kfcpp.get_prediction(dt))
    osd = np.array([obj1.sensor_data()]).T
    kf.update(osd)
    kfcpp.update(osd)

real_tract = np.array(real_tract)
track_from_sensor = np.array(track_from_sensor)
track_from_kf = np.array(track_from_kf)

# 3D Plot
# /////////////////////////////////////////
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(real_tract[:, 0], real_tract[:, 1], real_tract[:, 2], lw=2.0)
# ax.scatter(track_from_sensor[:, 0], track_from_sensor[:, 1], track_from_sensor[:, 2])
# ax.scatter(track_from_kf[:, 0], track_from_kf[:, 1], track_from_kf[:, 2], label="kalman")
# ax.plot(track_from_kf[:, 0], track_from_kf[:, 1], track_from_kf[:, 2])
# ax.set_xlabel("X Axis")
# ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
# plt.legend()
# plt.show()


# Filter error
# /////////////////////////////////////////
error_from_sensor = np.sum((real_tract - track_from_sensor) ** 2, axis=1) ** .5
error_from_filter = np.sum((real_tract - track_from_kf[:, 0:3]) ** 2, axis=1) ** .5
plt.plot(error_from_sensor + 1, label='sensor error')
plt.plot(error_from_filter + 1, label='filter error')
plt.yscale('log')
plt.grid()
plt.legend()
plt.show()
