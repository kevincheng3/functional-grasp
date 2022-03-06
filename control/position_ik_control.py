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
    fig, axs = plt.subplots(6, 1, sharex=True)
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)

    # Plot each graph, and manually set the y tick values
    # contact force
    axs[0].plot(sensor_data[:,24:29])
    axs[0].legend(["index", "mid","ring","little", "thu"], loc = 4, fontsize= 8)
    axs[0].text(0.1, 0.8, 'fingertip force', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes)# axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))

    # axs[0].set_title('fingertip force')# axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    # axs[0].set_ylim(-1, 1)

    axs[1].plot(sensor_data[:,3:7])
    # axs[1].set_yticks(np.arange(0.1, 1.0, 0.4))
    # axs[1].set_ylim(0, 1)
    axs[1].legend(["FFJ3", "FFJ2", "FFJ1", "FFJ0"],  loc = 4, fontsize= 8)
    axs[1].text(0.1, 0.8, 'index motor torque', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes)# axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))


    axs[2].plot(sensor_data[:,7:11])
    axs[2].legend(["MFJ3", "MFJ2", "MFJ1", "MFJ0"],  loc = 4, fontsize= 8)
    axs[2].text(0.1, 0.8, 'middle motor torque', horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes)# axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    # axs[2].set_yticks(np.arange(0.1, 1.0, 0.4))
    # axs[2].set_ylim(0, 1)

    axs[3].plot(sensor_data[:,11:15])
    axs[3].legend(["RFJ3", "RFJ2", "RFJ1", "RFJ0"],  loc = 4, fontsize= 8)
    axs[3].text(0.1, 0.8, 'ring motor torque', horizontalalignment='center', verticalalignment='center', transform=axs[3].transAxes)# axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    # axs[3].set_yticks(np.arange(0.1, 1.0, 0.4))
    # axs[3].set_ylim(0, 1)

    axs[4].plot(sensor_data[:,15:19])
    axs[4].legend(["LFJ3", "LFJ2", "LFJ1", "LFJ0"],  loc = 4, fontsize= 8)
    axs[4].text(0.1, 0.8, 'little motor torque', horizontalalignment='center', verticalalignment='center', transform=axs[4].transAxes)# axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    # axs[4].set_yticks(np.arange(0.1, 1.0, 0.4))
    # axs[4].set_ylim(0, 1)

    axs[5].plot(sensor_data[:,19:24])
    axs[5].legend(["THJ4", "THJ3", "THJ2", "THJ1", "THJ0"],  loc = 4, fontsize= 8)
    axs[5].text(0.1, 0.8, 'thumb motor torque', horizontalalignment='center', verticalalignment='center', transform=axs[5].transAxes)# axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))

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

    def apply_torque(self, joint_id, actuator_id, desired_pos, desired_vel, kp=20, kv=0.01 ):
        "pd controller"
        pos_torque = - kp * (self.sim.data.qpos[joint_id] - desired_pos)
        # print(self.sim.data.qpos[joint_id])
        vel_torque = - kv * (self.sim.data.qvel[joint_id] - desired_vel)

        self.sim.data.ctrl[actuator_id] = pos_torque + vel_torque


    def apply_torque_jacobian(self, id, desired_pos, desired_vel, fd_torque = 0, kp=20, kv=0.01 ):
        "fd_torque is generated by the contact force "
        pos_torque = - kp * ( - desired_pos)
        vel_torque = - kv * (self.sim.data.qvel[id+3] - desired_vel)

        self.sim.data.ctrl[id] = pos_torque + vel_torque + fd_torque


def main():
    data = np.load("functional grasp/data/data.npy")
    model = load_model_from_path("E:/zju/desktop_organization/functional grasp/assets/position_newhand.xml")
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
    niter = 4999

    # for i in range(5000):
    #     hand.apply_torque(hand.xr_joint_id, hand.xr_ac_id, -1.76, 0, kp=100)
    #     hand.apply_torque(hand.zr_joint_id, hand.zr_ac_id, 1.57, 0, kp=100)
    #     hand.apply_torque(hand.xs_joint_id, hand.xs_ac_id, 0, 0, kp=100)
    #     hand.apply_torque(hand.ys_joint_id, hand.ys_ac_id, -0.1, 0, kp=100)
    #     hand.apply_torque(hand.zs_joint_id, hand.zs_ac_id, 0, 0, kp=100)
    #     hand.apply_torque(hand.th1_joint_id, hand.th1_ac_id, data[11] / niter * i, 0, kp=100)
    #     sim.step()
    #     viewer.render()
    for i in range(2000):
        sim.data.ctrl[1] = -0.15
        sim.data.ctrl[2] = 0.1
        sim.data.ctrl[3] = -1.76
        sim.data.ctrl[5] = 1.57
        sim.step()
        viewer.render()

    for i in range(niter):

        sim.data.ctrl[:] = data[:] / niter * i
        sim.data.ctrl[1] = -0.15 + (data[1] + 0.15)/ niter * i
        sim.data.ctrl[2] = 0.1 + (data[2] - 0.1)/ niter * i
        sim.data.ctrl[3] = -1.76 + (data[3] + 1.76)/ niter * i        
        sim.data.ctrl[5] = 1.57 + (data[5]-1.57)/ niter * i
        sim.data.ctrl[10] = data[11] / niter * i
        sim.data.ctrl[11] = data[10] / niter * i
        # print(data)    
        # print(sim.data.ctrl[:]) 
        sim.step()
        viewer.render()
    for i in range(1000):

        sim.data.ctrl[:] = data[:]
        sim.data.ctrl[10] = data[11]
        sim.data.ctrl[11] = data[10]
        # print(data)    
        # print(sim.data.ctrl[:]) 
        sim.step()
        viewer.render()
    for i in range(8000):

        sim.data.ctrl[2] = sim.data.ctrl[2] + 0.00001
        # print(data)    
        # print(sim.data.ctrl[:]) 
        sim.step()
        viewer.render()
    print(sim.data.site_xpos[sim.model.site_name2id('robot0:ff_contact')])
    print(sim.data.site_xpos[sim.model.site_name2id('robot0:mf_contact')])
    print(sim.data.site_xpos[sim.model.site_name2id('robot0:rf_contact')])
    print(sim.data.site_xpos[sim.model.site_name2id('robot0:lf_contact')])
    print(sim.data.site_xpos[sim.model.site_name2id('robot0:th_contact')])

    print('____________________________________________________________________')
    print(sim.data.site_xmat[sim.model.site_name2id('robot0:ff_contact')].reshape(3,3).dot(np.array([1,0,0])))
    print(sim.data.site_xmat[sim.model.site_name2id('robot0:mf_contact')].reshape(3,3).dot(np.array([1,0,0])))
    print(sim.data.site_xmat[sim.model.site_name2id('robot0:rf_contact')].reshape(3,3).dot(np.array([1,0,0])))      
    print(sim.data.site_xmat[sim.model.site_name2id('robot0:lf_contact')].reshape(3,3).dot(np.array([1,0,0])))
    print(sim.data.site_xmat[sim.model.site_name2id('robot0:th_contact')].reshape(3,3).dot(np.array([0, -1, 0])))
        
if __name__ == '__main__':
    main()
