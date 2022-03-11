from mujoco_py import MjSim, MjViewer, load_model_from_path
import numpy as np
from scipy.spatial.transform import Rotation as R
import glfw
from mujoco_py import const
from enum import Enum
import matplotlib.pyplot as plt

def draw_picture(sensor_data):
    # t = np.arrange(0,5000,1)
    plt.style.use('fivethirtyeight')
    fig, axs = plt.subplots(2, 1, sharex=True)
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)

    # Plot each graph, and manually set the y tick values
    # contact force
    axs[0].plot(sensor_data[:,6:11])
    axs[0].legend(["index", "middle","ring","little", "thumb"], loc = 1, fontsize= 15)
    axs[0].text(0.2, 0.8, 'fingertip force', fontsize=20, horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)# axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))

    # axs[0].set_title('fingertip force')# axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    # axs[0].set_ylim(-1, 1)

    axs[1].plot(sensor_data[:,0:6])
    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.4))
    # axs[1].set_ylim(0, 1)
    axs[1].legend(["index", "middle", "ring", "little", "thumb1", "thumb2"],  loc = 1, fontsize= 15)
    axs[1].text(0.2, 0.8, 'finger motor torque', fontsize=20, horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)# axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))


    plt.show()


class Controller():
    def __init__(self, sim) -> None:
        super().__init__()
        self.sim = sim
        self.xs_joint_id = sim.model.joint_name2id('robot0:root_x')
        self.xs_ac_id = sim.model.actuator_name2id('robot0:root_x')

        self.ys_joint_id = sim.model.joint_name2id('robot0:root_y')
        self.ys_ac_id = sim.model.actuator_name2id('robot0:root_y')

        self.zs_joint_id = sim.model.joint_name2id('robot0:root_z')
        self.zs_ac_id = sim.model.actuator_name2id('robot0:root_z')

        self.xr_joint_id = sim.model.joint_name2id('robot0:root_rotationx')
        self.xr_ac_id = sim.model.actuator_name2id('robot0:root_rotationx')

        self.yr_joint_id = sim.model.joint_name2id('robot0:root_rotationy')
        self.yr_ac_id = sim.model.actuator_name2id('robot0:root_rotationy')

        self.zr_joint_id = sim.model.joint_name2id('robot0:root_rotationz')
        self.zr_ac_id = sim.model.actuator_name2id('robot0:root_rotationz')

        self.ff_joint_id = sim.model.joint_name2id('robot0:FFJ2')
        self.ff_ac_id = sim.model.actuator_name2id('robot0:A_FFJ2')

        self.mf_joint_id = sim.model.joint_name2id('robot0:MFJ2')
        self.mf_ac_id = sim.model.actuator_name2id('robot0:A_MFJ2')

        self.rf_joint_id = sim.model.joint_name2id('robot0:RFJ2')
        self.rf_ac_id = sim.model.actuator_name2id('robot0:A_RFJ2')

        self.lf_joint_id = sim.model.joint_name2id('robot0:LFJ2')
        self.lf_ac_id = sim.model.actuator_name2id('robot0:A_LFJ2')

        self.th2_joint_id = sim.model.joint_name2id('robot0:THJ3')
        self.th2_ac_id = sim.model.actuator_name2id('robot0:A_THJ3')

        self.th1_joint_id = sim.model.joint_name2id('robot0:THJ2')
        self.th1_ac_id = sim.model.actuator_name2id('robot0:A_THJ2')

    def apply_torque(self, joint_id, actuator_id, desired_pos, desired_vel, fd_torque = 0, kp = 100, kv = 0.01 ):
        "pd controller"
        pos_torque = - kp * (self.sim.data.qpos[joint_id] - desired_pos)
        # print(self.sim.data.qpos[joint_id])
        vel_torque = - kv * (self.sim.data.qvel[joint_id] - desired_vel)

        self.sim.data.ctrl[actuator_id] = pos_torque + vel_torque + fd_torque
        # if joint_id == self.ff_joint_id:
        #     print('pos_torque', pos_torque, 'fd_torque', fd_torque)

def main():
    data = np.load("functional grasp/data/data.npy")
    model = load_model_from_path("E:/zju/desktop_organization/functional grasp/assets/newhand.xml")
    sim = MjSim(model)
    # viewer set up
    viewer = MjViewer(sim)
    hand = Controller(sim)
    body_id = sim.model.body_name2id('robot0:hand mount')
    lookat = sim.data.body_xpos[body_id]
    for idx, value in enumerate(lookat):
        viewer.cam.lookat[idx] = value
    viewer.cam.distance = 2
    viewer.cam.azimuth = 180.
    viewer.cam.elevation = -30
    viewer._paused = 1
    niter = 6999
    sensor_data=[]

    # for i in range(niter):
    #     hand.apply_torque(hand.xr_joint_id, hand.xr_ac_id, 0, 0)
    #     hand.apply_torque(hand.zr_joint_id, hand.zr_ac_id, 0, 0)
    #     hand.apply_torque(hand.xs_joint_id, hand.xs_ac_id, 0, 0)
    #     hand.apply_torque(hand.ys_joint_id, hand.ys_ac_id, 0, 0)
    #     hand.apply_torque(hand.zs_joint_id, hand.zs_ac_id, 0, 0, kp=10000)
    #     hand.apply_torque(hand.th1_joint_id, hand.th1_ac_id, 0, 0)
    #     hand.apply_torque(hand.ff_joint_id, hand.ff_ac_id, 0, 0)
    #     hand.apply_torque(hand.mf_joint_id, hand.mf_ac_id, 0, 0)
    #     hand.apply_torque(hand.rf_joint_id, hand.rf_ac_id, 0, 0)
    #     hand.apply_torque(hand.lf_joint_id, hand.lf_ac_id, 0, 0)
    #     hand.apply_torque(hand.th2_joint_id, hand.th2_ac_id, 0, 0)
    #     sim.step()
    #     viewer.render()

    # print(sim.data.site_xpos)

    for i in range(1000):
        hand.apply_torque(hand.xr_joint_id, hand.xr_ac_id, -1.76, 0)
        hand.apply_torque(hand.zr_joint_id, hand.zr_ac_id, 1.57, 0)
        hand.apply_torque(hand.xs_joint_id, hand.xs_ac_id, 0, 0)
        hand.apply_torque(hand.ys_joint_id, hand.ys_ac_id, -0.15, 0)
        hand.apply_torque(hand.zs_joint_id, hand.zs_ac_id, 0, 0, kp=10000)
        hand.apply_torque(hand.th1_joint_id, hand.th1_ac_id, data[11] / niter * i, 0)
        hand.apply_torque(hand.ff_joint_id, hand.ff_ac_id, 0, 0)
        hand.apply_torque(hand.mf_joint_id, hand.mf_ac_id, 0, 0)
        hand.apply_torque(hand.rf_joint_id, hand.rf_ac_id, 0, 0)
        hand.apply_torque(hand.lf_joint_id, hand.lf_ac_id, 0, 0)
        hand.apply_torque(hand.th2_joint_id, hand.th2_ac_id, 0, 0)
        # sensor_data.append(sim.data.sensordata.copy())
        sim.step()
        # viewer.render()

    for i in range(niter + 1):
        hand.apply_torque(hand.xs_joint_id, hand.xs_ac_id, data[0] / niter * i, 0)      
        hand.apply_torque(hand.ys_joint_id, hand.ys_ac_id, -0.15 + (data[1] + 0.15) / niter * i, 0)
        hand.apply_torque(hand.zs_joint_id, hand.zs_ac_id, data[2] / niter * i, 0, kp=10000)
        hand.apply_torque(hand.xr_joint_id, hand.xr_ac_id, -1.76 + (data[3] + 1.76)/ niter * i, 0)
        hand.apply_torque(hand.yr_joint_id, hand.yr_ac_id, data[4] / niter * i, 0, kp=1000)
        hand.apply_torque(hand.zr_joint_id, hand.zr_ac_id, 1.57 + (data[5] - 1.57)/ niter * i, 0)
        hand.apply_torque(hand.ff_joint_id, hand.ff_ac_id, data[6] / niter * i, 0)
        hand.apply_torque(hand.mf_joint_id, hand.mf_ac_id, data[7] / niter * i, 0)
        hand.apply_torque(hand.rf_joint_id, hand.rf_ac_id, data[8] / niter * i, 0)
        hand.apply_torque(hand.lf_joint_id, hand.lf_ac_id, data[9] / niter * i, 0)
        hand.apply_torque(hand.th2_joint_id, hand.th2_ac_id, data[10] / niter * i, 0)
        hand.apply_torque(hand.th1_joint_id, hand.th1_ac_id, data[11], 0, kp=20)
        sim.step()
        viewer.render()

    for i in range(1000):
        hand.apply_torque(hand.xs_joint_id, hand.xs_ac_id, data[0], 0)      
        hand.apply_torque(hand.ys_joint_id, hand.ys_ac_id, data[1], 0)
        hand.apply_torque(hand.zs_joint_id, hand.zs_ac_id, data[2], 0)
        hand.apply_torque(hand.xr_joint_id, hand.xr_ac_id, data[3], 0)
        hand.apply_torque(hand.yr_joint_id, hand.yr_ac_id, data[4], 0)
        hand.apply_torque(hand.zr_joint_id, hand.zr_ac_id, data[5], 0)
        hand.apply_torque(hand.ff_joint_id, hand.ff_ac_id, data[6], 0)
        hand.apply_torque(hand.mf_joint_id, hand.mf_ac_id, data[7], 0)
        hand.apply_torque(hand.rf_joint_id, hand.rf_ac_id, data[8], 0)
        hand.apply_torque(hand.lf_joint_id, hand.lf_ac_id, data[9], 0)
        hand.apply_torque(hand.th2_joint_id, hand.th2_ac_id, data[10], 0)
        hand.apply_torque(hand.th1_joint_id, hand.th1_ac_id, data[11], 0)
        sensor_data.append(sim.data.sensordata.copy())
        sim.step()
        viewer.render()

    for i in range(5000):
        hand.apply_torque(hand.xs_joint_id, hand.xs_ac_id, data[0], 0)
        hand.apply_torque(hand.ys_joint_id, hand.ys_ac_id, data[1], 0)
        hand.apply_torque(hand.zs_joint_id, hand.zs_ac_id, data[2] + 0.0001 * i, 0)
        hand.apply_torque(hand.xr_joint_id, hand.xr_ac_id, data[3], 0)
        hand.apply_torque(hand.yr_joint_id, hand.yr_ac_id, data[4], 0)
        hand.apply_torque(hand.zr_joint_id, hand.zr_ac_id, data[5], 0)
        hand.apply_torque(hand.ff_joint_id, hand.ff_ac_id, data[6], 0, fd_torque= 0)
        hand.apply_torque(hand.mf_joint_id, hand.mf_ac_id, data[7], 0, fd_torque= 0)
        hand.apply_torque(hand.rf_joint_id, hand.rf_ac_id, data[8], 0, fd_torque= 0)
        hand.apply_torque(hand.lf_joint_id, hand.lf_ac_id, data[9], 0, fd_torque= 0)
        hand.apply_torque(hand.th2_joint_id, hand.th2_ac_id, data[10], 0)
        hand.apply_torque(hand.th1_joint_id, hand.th1_ac_id, data[11], 0, fd_torque= 0)
        sensor_data.append(sim.data.sensordata.copy())
        sim.step()
        viewer.render()
    # print(sim.data.site_xpos[sim.model.site_name2id('robot0:ff_contact')])
    # print(sim.data.site_xmat[sim.model.site_name2id('robot0:ff_contact')].reshape(3,3).dot(np.array([1,0,0])))
    
    # print(sim.data.site_xpos[sim.model.site_name2id('robot0:mf_contact')])
    # print(sim.data.site_xmat[sim.model.site_name2id('robot0:mf_contact')].reshape(3,3).dot(np.array([1,0,0])))
    
    # print(sim.data.site_xpos[sim.model.site_name2id('robot0:rf_contact')])
    # print(sim.data.site_xmat[sim.model.site_name2id('robot0:rf_contact')].reshape(3,3).dot(np.array([1,0,0])))
                    
    # print(sim.data.site_xpos[sim.model.site_name2id('robot0:lf_contact')])
    # print(sim.data.site_xmat[sim.model.site_name2id('robot0:lf_contact')].reshape(3,3).dot(np.array([1,0,0])))

    # print(sim.data.site_xpos[sim.model.site_name2id('robot0:th_contact')])
    # print(sim.data.site_xmat[sim.model.site_name2id('robot0:th_contact')].reshape(3,3).dot(np.array([0, -1, 0])))
    sensor_data = np.array(sensor_data)
    draw_picture(sensor_data)
if __name__ == '__main__':
    main()
