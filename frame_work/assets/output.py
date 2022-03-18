import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List
import os
import glob
import random
from ruamel.yaml import YAML

ASSETS_DIR = "E:/zju/desktop_organization/functional grasp/frame_work/assets" 
def find_obj_meshes() -> Dict[str, list]:
    return find_meshes_by_dirname("obj")

# all of the object mesh
def find_meshes_by_dirname(root_mesh_dir) -> Dict[str, list]:
    """
    Find all meshes under given mesh directory, grouped by top level
    folder name.
    :param root_mesh_dir: The root directory name for mesh files.
    :return: {dir_name -> list of mesh files}
    """
    root_path = os.path.join(ASSETS_DIR, root_mesh_dir)

    all_stls = {}
    all_file = []
    for subdir in os.listdir(root_path):
        # print(subdir)
        all_file.append(subdir)
        curr_path = os.path.join(root_path, subdir)
        if not os.path.isdir(curr_path) and not curr_path.endswith(".stl"):
            continue

        if curr_path.endswith(".stl"):
            stls = [curr_path]
        else:
            stls = glob.glob(os.path.join(curr_path, "*.stl"))
        assert len(stls) > 0
        assert len(all_file) > 0

        all_stls[subdir] = stls

    assert len(all_stls) > 0
    return all_stls, all_file

def make_mesh_object(name: str, files: List[str], scale: float):
    # Center mesh properly by offsetting with center position of combined mesh.
    # x,y,z are the spatial coordinates of the site respectively
    # print(pos_string)
    # print(files[0].replace('obj', 'visual'))    
    visual_file = files[0].replace('obj', 'visual')
    scale_string = " ".join(map(str, [scale] * 3))
    assets = [
        f'<mesh file="{file}" name="{name}-{idx}" scale="{scale_string}" />'
        for idx, file in enumerate(files)
    ]
    assets.append(f'<mesh file="{visual_file}" name="{name}" scale="0.001 0.001 0.001" />') # the visual part of assets

    geoms = [
        f'<geom type="mesh" class="robot0:DC_obj" mesh="{name}-{idx}" pos="0 0 0" />'
        for idx in range(len(files))
    ]
    geoms.append(f'<geom type="mesh" mesh="{name}" class="robot0:D_Vizual" pos="0 0 0" rgba = "0.5 0.5 0.5 0.8"/>') 

    assets_xml = "\n".join(assets)
    geoms_xml = "\n".join(geoms)
    # print(geoms_xml)
    xml_source = f"""
    <mujoco>
      <asset>
        {assets_xml}
      </asset>
      <worldbody>
        <body name="{name}" pos="0 0 .00">
          {geoms_xml}
            <joint name="{name}:joint" type="free"/>
        </body>
      </worldbody>
    </mujoco>
    """
    return ET.fromstring(xml_source)

curr_dir = os.path.dirname(os.path.abspath(__file__))
tree = ET.parse(curr_dir+'/basic.xml')

MESH_FILES, OBJ_FILES = find_obj_meshes()
candidates = list(MESH_FILES.values())
candidates = sorted(candidates)
objs = sorted(OBJ_FILES)
last_seed = (random.randint(0, 2 ** 32 - 1))

# choose one object randomly

indices = np.random.RandomState(last_seed).choice(
            len(candidates), size= 1, replace=False)

print('the object selected is:', objs[indices[0]])

# decide the way to handle object
print("please enter a command(0 for handoff, 1 for use)")
way = ["handoff", "use"]

i = int(input())

# get the parameter

cfg_path = "functional grasp/frame_work/config/para.yaml"
cfg = YAML().load(open(cfg_path, 'r'))

contact_site = []
contact_site.append(cfg[str(objs[indices[0]])][way[i]]["thumb"])
contact_site.append(cfg[str(objs[indices[0]])][way[i]]["index"])
contact_site.append(cfg[str(objs[indices[0]])][way[i]]["middle"])
contact_site.append(cfg[str(objs[indices[0]])][way[i]]["ring"])
contact_site.append(cfg[str(objs[indices[0]])][way[i]]["little"])

contact_site.append(cfg[str(objs[indices[0]])][way[i]]["thumb_vec"])
contact_site.append(cfg[str(objs[indices[0]])][way[i]]["index_vec"])
contact_site.append(cfg[str(objs[indices[0]])][way[i]]["middle_vec"])
contact_site.append(cfg[str(objs[indices[0]])][way[i]]["ring_vec"])
contact_site.append(cfg[str(objs[indices[0]])][way[i]]["little_vec"])

contact_para = np.array(contact_site).ravel()
print(contact_para)

constraint = []
constraint.append(cfg['constraint'][str(objs[indices[0]])]["upper bound"])
constraint.append(cfg['constraint'][str(objs[indices[0]])]["lower bound"])
constraint_para = np.array(constraint)

ini = []
ini.append(cfg['ini'][str(objs[indices[0]])])
x0 = np.array(ini)

# optimize the hand conguration

import numpy as np
from scipy.optimize import minimize
from math import cos, sin, atan
import argparse
from scipy.spatial.transform import Rotation as R

def rotation(theta_x=0, theta_y=0, theta_z=0):

    rot_x = np.array([[1, 0, 0],[0, np.cos(theta_x), - np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],[0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    rot_z = np.array([[np.cos(theta_z), - np.sin(theta_z), 0],[ np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    R = rot_x.dot(rot_y).dot(rot_z)

    return R

def quat2euler(quat):
    # transfer quat to euler
    r = R.from_quat(np.array([quat[1], quat[2], quat[3], quat[0]]))
    return r.as_euler('XYZ')

class Hand(object):
    def __init__(self):

        self.L00x = -0.00135 # extend direction  <body ffknuckle pos[2]>
        self.L00y = 0.03919 # bias direction   <body ffknuckle pos[0]>
        self.L00z = 0.13556 # bias direction   <body ffknuckle pos[0]>
        self.L01 = 0.046   # <body ffmiddle pos[0]>
        self.L02 = 0.023   # <body ffdistal pos[0]>
        self.L03x = 0.0055
        self.L03z = 0.02

        # middle
        self.L10x = -0.00128
        self.L10y = 0.01520
        self.L10z = 0.13822
        self.L11 = 0.05
        self.L12 = 0.03
        self.L13x = 0.0055
        self.L13z = 0.018

        # ring
        self.L20x = -0.00121 
        self.L20y = -0.01057 
        self.L20z = 0.13759
        self.L21 = 0.048
        self.L22 = 0.029
        self.L23x = 0.0055
        self.L23z = 0.02

        # little
        self.L30x = -0.00114
        self.L30y = -0.03517
        self.L30z = 0.13459
        self.L31 = 0.04
        self.L32 = 0.021
        self.L33x = 0.0055
        self.L33z = 0.02

        # thumb
        self.L40x = 0.005 # extend direction
        self.L40y = 0.02
        self.L40z = 0.044 # bias direction

        self.L41y = 0.056
        self.L41z = 0.03697

        self.L42 = 0.039

        self.L43y = -0.007
        self.L43z = 0.022

    def rotation(self, theta_x, theta_y, theta_z):
        rot_x = np.array([[1, 0, 0],[0, np.cos(theta_x), - np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
        rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],[0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
        rot_z = np.array([[np.cos(theta_z), - np.sin(theta_z), 0],[ np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
        R = rot_x.dot(rot_y).dot(rot_z)

        return R
      
    def fingertip_pos(self, pos):
        Rw = R.from_euler("XYZ", [0, 0, 0]).as_matrix() # base body pose
        pose = pos[3:6] # rotation of the palm
        matrix = R.from_euler("XYZ", pose).as_matrix() # rotation of the palm

        R0 = np.array(Rw).dot(np.array(matrix))

        v1 = np.array([0, 0, 0.2])
        v2 =  np.array(pos[0:3])
        pos0 = np.array(Rw).dot(v2)

        palm_pos = v1 + pos0

        "--------------------------------------------------------"

        ff_ovec = np.array([1, 0, 0])

        r00 = R.from_euler('x', -0.0395)
        r01 = R.from_euler('y', pos[6])
        r02 = R.from_euler('y', pos[6] * 7094 / 5806.0)
        r03 = R.from_euler('y', pos[6] * 3996 / 5806.0)

        R01 = r00 * r01
        R02 = R01 * r02 
        R03 = R02 * r03

        v00 = np.array([self.L00x, self.L00y, self.L00z])
        v01 = [0, 0, self.L01]
        v02 = [0, 0, self.L02]
        v03 = [self.L03x, 0, self.L03z]   

        ff_pos1 =  R01.apply(v01)
        ff_pos2 =  R02.apply(v02)
        ff_pos3 =  R03.apply(v03)
        # pos = pos1 + pos2 + pos3
        ff_pos = ff_pos1 + ff_pos2 + ff_pos3 + v00
        # print((self.R0))
        ff_vec = R0.dot(np.array(R03.as_matrix())).dot(ff_ovec)
        # print(pos1, pos2, pos3)
        ff_pos = R0.dot(ff_pos) + palm_pos
        "--------------------------------------------------------"

        mf_ovec = np.array([1, 0, 0])

        r10 = R.from_euler('x', 0)
        r11 = R.from_euler('y', pos[7])
        r12 = R.from_euler('y', pos[7] * 7094 / 5806.0)
        r13 = R.from_euler('y', pos[7] * 3996 / 5806.0)

        R11 = r10 * r11
        R12 = R11 * r12 
        R13 = R12 * r13
        # print(R11.as_matrix())
        # print(R12.as_matrix())
        # print(R13.as_matrix())

        v10 = np.array([self.L10x, self.L10y, self.L10z])
        v11 = [0, 0, self.L11]
        v12 = [0, 0, self.L12]
        v13 = [self.L13x, 0, self.L13z]   

        mf_pos1 =  R11.apply(v11)
        mf_pos2 =  R12.apply(v12)
        mf_pos3 =  R13.apply(v13)
        # pos = pos1 + pos2 + pos3
        mf_pos = mf_pos1 + mf_pos2 + mf_pos3 + v10
        
        # print((self.R0))
        mf_vec = R0.dot(np.array(R13.as_matrix())).dot(mf_ovec)
        # print(pos1, pos2, pos3)
        mf_pos = R0.dot(mf_pos) + palm_pos

        "--------------------------------------------------------"

        rf_ovec = np.array([1, 0, 0])

        r20 = R.from_euler('x', 0.0726)
        r21 = R.from_euler('y', pos[8])
        r22 = R.from_euler('y', pos[8] * 7094 / 5806.0)
        r23 = R.from_euler('y', pos[8] * 3996 / 5806.0)

        R21 = r20 * r21
        R22 = R21 * r22 
        R23 = R22 * r23

        v20 = np.array([self.L20x, self.L20y, self.L20z])
        v21 = [0, 0, self.L21]
        v22 = [0, 0, self.L22]
        v23 = [self.L23x, 0, self.L23z]   

        rf_pos1 =  R21.apply(v21)
        rf_pos2 =  R22.apply(v22)
        rf_pos3 =  R23.apply(v23)
        # pos = pos1 + pos2 + pos3
        rf_pos = rf_pos1 + rf_pos2 + rf_pos3 + v20
        # print((self.R0))
        rf_vec = R0.dot(np.array(R23.as_matrix())).dot(rf_ovec)
        # print(pos1, pos2, pos3)
        rf_pos = R0.dot(rf_pos) + palm_pos

        "--------------------------------------------------------"

        lf_ovec = np.array([1, 0, 0])

        r30 = R.from_euler('x', 0.108)
        r31 = R.from_euler('y', pos[9])
        r32 = R.from_euler('y', pos[9] * 7094 / 5806.0)
        r33 = R.from_euler('y', pos[9] * 3996 / 5806.0)

        R31 = r30 * r31
        R32 = R31 * r32 
        R33 = R32 * r33

        v30 = np.array([self.L30x, self.L30y, self.L30z])
        v31 = [0, 0, self.L31]
        v32 = [0, 0, self.L32]
        v33 = [self.L33x, 0, self.L33z]   

        lf_pos1 =  R31.apply(v31)
        lf_pos2 =  R32.apply(v32)
        lf_pos3 =  R33.apply(v33)
        # pos = pos1 + pos2 + pos3
        lf_pos = lf_pos1 + lf_pos2 + lf_pos3 + v30
        # print((self.R0))
        lf_vec = R0.dot(np.array(R33.as_matrix())).dot(lf_ovec)
        # print(pos1, pos2, pos3)
        lf_pos = R0.dot(lf_pos) + palm_pos

        "--------------------------------------------------------"

        th_ovec = np.array([0, -1, 0])

        r40 = R.from_euler('z', -pos[10])
        r41 = R.from_euler('x', -0.65)
        r42 = R.from_euler('x', pos[11])
        r43 = R.from_euler('x', pos[11] * 6921 / 7218)

        R41 = r40
        R42 = R41 * r41 * r42
        R43 = R42 * r43

        v40 = np.array([self.L40x, self.L40y, self.L40z])
        v41 = [0, self.L41y, self.L41z]
        v42 = [0, 0, self.L42]
        v43 = [0, self.L43y, self.L43z]

        th_pos1 =  R41.apply(v41)
        th_pos2 =  R42.apply(v42)
        th_pos3 =  R43.apply(v43)
        # pos = pos1 + pos2 + pos3
        th_pos = th_pos1 + th_pos2 + th_pos3 + v40
        # print((self.R0))
        th_vec = R0.dot(np.array(R43.as_matrix())).dot(th_ovec)
        # print(pos1, pos2, pos3)
        th_pos = R0.dot(th_pos) + palm_pos

        pos = np.concatenate((th_pos, ff_pos, mf_pos, rf_pos, lf_pos))
        vec = np.concatenate((th_vec, ff_vec, mf_vec, rf_vec, lf_vec))
        return pos, vec

def cost(x, infor_object):

    """
    x[0:3] : palm pos
    x[3:6] : palm orientation
    x[6:10] : index theta
    x[10:14] : middle theta
    x[14:18] : ring theta
    x[18:22] : little theta
    x[22:27] : thumb theta
    """
    n = int(len(infor_object) / 2)
    obj_pos = infor_object[:n]
    obj_vec = infor_object[n:]
    # obj_pos = infor_object

    hand = Hand()

    assert len(infor_object) % 6 == 0, "the dimention of the object is incorrect"
    hand_pos, hand_vec = hand.fingertip_pos(x[:])    
    pos_cost = np.linalg.norm(1000 * (hand_pos[:] - np.array( obj_pos[:])))
                
    vec_cost = 10 * np.dot(np.array( hand_vec[:]), np.array( obj_vec[:]))
          
    return pos_cost + vec_cost
    # return pos_cost

class Object(object):
    def __inint__(self, pos_thumb, pos_index, pos_middle, pos_ring, pos_little, vec_thumb, vec_index, vec_middle, vec_ring, vec_little):

        self.thumb_contact = pos_thumb
        self.index_contact = pos_index
        self.middle_contact = pos_middle
        self.ring_contact = pos_ring
        self.little_contact = pos_little

        self.thumb_vec = vec_thumb
        self.index_vec = vec_index
        self.middle_vec = vec_middle
        self.ring_vec = vec_ring
        self.thumb_vec = vec_little

    def get_infor(self):
        return self.thumb_contact, self.index_contact, self.middle_contact, self.ring_contact,self.little_contact

from scipy.optimize import Bounds

bounds = Bounds(constraint_para[1], constraint_para[0])

res = minimize(cost, x0, args=(contact_para,),method = "L-BFGS-B", 
            options={'verbose': 1}, bounds=bounds)
# res.x = np.zeros(12)
np.save("functional grasp/frame_work/data/result.npy", res.x)   
# print(res.x)
hand = Hand()
hand_pos, hand_vec = hand.fingertip_pos(res.x) 
print(hand_pos)
print("-----------------------")
print()



# write the xml for the grasp
xmls = []

object_xml = make_mesh_object(
    f"object",
    candidates[indices[0]],
    scale=0.1
)

xmls.append(object_xml)
root = tree.getroot()   

other = ET.ElementTree(xmls[0])
root.extend(other.getroot())

Output_dir = "E:/zju/desktop_organization/functional grasp/frame_work/assets/output.xml"
xml_string = ET.tostring(root, encoding="unicode", method="xml")
ET.ElementTree(root).write(Output_dir)

# visual and control

from mujoco_py import MjSim, MjViewer, load_model_from_path
import numpy as np
from scipy.spatial.transform import Rotation as R

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

data = res.x
# data = np.load("functional grasp/frame_work/data/result.npy")
# data = np.load("functional grasp/data/wine_glass/data.npy")

# data = np.load("functional grasp/data/data.npy")
model = load_model_from_path(Output_dir)
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

if str(objs[indices[0]]) == "stapler":
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

        sim.step()
        viewer.render()

    for i in range(5000):
        hand.apply_torque(hand.xs_joint_id, hand.xs_ac_id, data[0], 0)
        hand.apply_torque(hand.ys_joint_id, hand.ys_ac_id, data[1], 0)
        hand.apply_torque(hand.zs_joint_id, hand.zs_ac_id, data[2] + 0.0001 * i, 0)
        hand.apply_torque(hand.xr_joint_id, hand.xr_ac_id, data[3], 0)
        hand.apply_torque(hand.yr_joint_id, hand.yr_ac_id, data[4], 0)
        hand.apply_torque(hand.zr_joint_id, hand.zr_ac_id, data[5], 0)
        hand.apply_torque(hand.ff_joint_id, hand.ff_ac_id, data[6], 0, fd_torque= 0.1)
        hand.apply_torque(hand.mf_joint_id, hand.mf_ac_id, data[7], 0, fd_torque= 0.1)
        hand.apply_torque(hand.rf_joint_id, hand.rf_ac_id, data[8], 0, fd_torque= 0.1)
        hand.apply_torque(hand.lf_joint_id, hand.lf_ac_id, data[9], 0, fd_torque= 0.1)
        hand.apply_torque(hand.th2_joint_id, hand.th2_ac_id, data[10], 0)
        hand.apply_torque(hand.th1_joint_id, hand.th1_ac_id, data[11], 0, fd_torque= 0.1)
        sensor_data.append(sim.data.sensordata.copy())
        sim.step()
        viewer.render()


if str(objs[indices[0]]) == "wine_glass":
    for i in range(1000):
        hand.apply_torque(hand.xs_joint_id, hand.xs_ac_id, 0.02, 0)
        hand.apply_torque(hand.ys_joint_id, hand.ys_ac_id, -0.2, 0)
        hand.apply_torque(hand.zs_joint_id, hand.zs_ac_id, 0.1, 0, kp=10000)        
        hand.apply_torque(hand.xr_joint_id, hand.xr_ac_id, -1.41, 0)
        hand.apply_torque(hand.zr_joint_id, hand.zr_ac_id, 3.14, 0)
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
        hand.apply_torque(hand.xs_joint_id, hand.xs_ac_id, 0.02 + (data[0] - 0.02) / niter * i, 0)      
        hand.apply_torque(hand.ys_joint_id, hand.ys_ac_id, -0.2 + (data[1] + 0.2) / niter * i, 0)
        hand.apply_torque(hand.zs_joint_id, hand.zs_ac_id, 0.1 + (data[2] - 0.1) / niter * i, 0, kp=10000)
        hand.apply_torque(hand.xr_joint_id, hand.xr_ac_id, -1.41 + (data[3] + 1.41)/ niter * i, 0)
        hand.apply_torque(hand.yr_joint_id, hand.yr_ac_id, data[4] / niter * i, 0, kp=1000)
        hand.apply_torque(hand.zr_joint_id, hand.zr_ac_id, 3.14 + (data[5] - 3.14)/ niter * i, 0)
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
        hand.apply_torque(hand.ff_joint_id, hand.ff_ac_id, data[6] + 0.2, 0, fd_torque= 0.1)
        hand.apply_torque(hand.mf_joint_id, hand.mf_ac_id, data[7] + 0.2, 0, fd_torque= 0.1)
        hand.apply_torque(hand.rf_joint_id, hand.rf_ac_id, data[8] + 0.2, 0, fd_torque= 0.1)
        hand.apply_torque(hand.lf_joint_id, hand.lf_ac_id, data[9] + 0.2, 0, fd_torque= 0.1)
        hand.apply_torque(hand.th2_joint_id, hand.th2_ac_id, data[10], 0)
        hand.apply_torque(hand.th1_joint_id, hand.th1_ac_id, data[11] + 0.2, 0, fd_torque= 0.1)
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
        hand.apply_torque(hand.ff_joint_id, hand.ff_ac_id, data[6] + 0.2, 0, fd_torque= 0.1)
        hand.apply_torque(hand.mf_joint_id, hand.mf_ac_id, data[7] + 0.2, 0, fd_torque= 0.1)
        hand.apply_torque(hand.rf_joint_id, hand.rf_ac_id, data[8] + 0.2, 0, fd_torque= 0.1)
        hand.apply_torque(hand.lf_joint_id, hand.lf_ac_id, data[9] + 0.2, 0, fd_torque= 0.1)
        hand.apply_torque(hand.th2_joint_id, hand.th2_ac_id, data[10], 0)
        hand.apply_torque(hand.th1_joint_id, hand.th1_ac_id, data[11] + 0.2, 0, fd_torque= 0.1)
        sensor_data.append(sim.data.sensordata.copy())
        sim.step()
        viewer.render()