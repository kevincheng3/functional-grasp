from mujoco_py import MjSim, MjViewer, load_model_from_path
import numpy as np
import matplotlib.pyplot as plt

model = load_model_from_path("E:/zju/desktop_organization/functional grasp/assets/contact_force.xml")
sim = MjSim(model)
# viewer set up
viewer = MjViewer(sim)
sensor_data=[]
for i in range(3000):  
    sim.data.ctrl[0] = 5e1
    sim.data.ctrl[1] = -5e1
    sim.step()
    viewer.render()
    sensor_data.append(sim.data.sensordata.copy())
for i in range(200000):
    sim.data.ctrl[0] = 5e1
    sim.data.ctrl[1] = -5e1
    sim.data.ctrl[2:4]  = 20
    sim.step()
    viewer.render()
    sensor_data.append(sim.data.sensordata.copy())
sensor_data = np.array(sensor_data)
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(sensor_data[:,0])
axs[1].plot(sensor_data[:,1])
print(sim.data.sensordata.copy())

plt.show()