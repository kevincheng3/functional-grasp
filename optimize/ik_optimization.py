import numpy as np
from scipy.optimize import minimize
from math import cos, sin, atan
import argparse
from scipy.spatial.transform import Rotation as R

# target: 
# 1. the position and orientation of the palm
# 2. the angle of the 

# input:
# 1. the contact position and the normal direction on the object 


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

parser = argparse.ArgumentParser()
parser.add_argument("--theta0", help="angle of theta0", type = float, default = 0.0)
parser.add_argument("--theta1", help="angle of theta1", type = float, default = 0.0)
parser.add_argument("--theta2", help="angle of theta2", type = float, default = 0.0)
parser.add_argument("--theta3", help="angle of theta3", type = float, default = 0.0)
parser.add_argument("--theta4", help="angle of theta4", type = float, default = 0.0)

args = parser.parse_args()
theta0 = args.theta0
theta1 = args.theta1
theta2 = args.theta2
theta3 = args.theta3
theta4 = args.theta4

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

        pos = np.concatenate((ff_pos, mf_pos, rf_pos, lf_pos, th_pos))
        vec = np.concatenate((ff_vec, mf_vec, rf_vec, lf_vec, th_vec))
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

    hand = Hand()

    assert len(infor_object) % 6 == 0, "the dimention of the object is incorrect"
    hand_pos, hand_vec = hand.fingertip_pos(x[:])    

    pos_cost = np.linalg.norm(1000 * (hand_pos[:] - np.array( obj_pos[:])))
                
    vec_cost = 10 * np.dot(np.array( hand_vec[:]), np.array( obj_vec[:]))
    
    # print("hand:", hand_pos, '\n', hand_vec)  
    # print(pos_cost, vec_cost)
    return pos_cost + vec_cost


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

def collision_constraint():
    # the kinematics constraint of the hand
    # 1.contact position 
    # 2.normal direction 
    # 3.fingertip position 
    # 4.fingertip orientation
    pass

from scipy.optimize import Bounds

bounds = Bounds([-0.25, 0.0, -0.3, -0.75, -0.75, -0.75, -0.436, 0, 0,    0, -1.047, 0, -0.262, -0.524, -1.571], 
                [0.25, 0.2, 0.5, 0.75, 0.75, 0.75, 0.436,  1.571, 1.571, 1.571, 1.047, 1.309, 0.262, 0.524, 0])

def main():
    # pos = np.zeros(12)
    # hand = Hand()
    # position, vector = hand.fingertip_pos(pos)
    # for i in range(0,15,3):
    #     print(position[i:i+3] )
    #     print(vector[i:i+3])
    constraint_para = np.load("../data/constraint_data.npy")

    contact_para = np.load('../data/all_contact_data.npy') 
    # print('constraint', constraint_para[0].shape[0])
    # print('contact', contact_para)
    bounds = Bounds(constraint_para[0][:], constraint_para[1][:])
    x0 = np.load("../data/ini.npy")

    res = minimize(cost, x0, args=(contact_para,),method = "L-BFGS-B", 
               options={'verbose': 1}, bounds=bounds)
    # res.x = np.zeros(12)
    np.save("../data/data.npy", res.x)   
    # print(res.x)
    hand = Hand()
    hand_pos, hand_vec = hand.fingertip_pos(res.x) 
    for i in range(0,15,3):
        print(hand_pos[i:i+3])

    print('____________________________________________________________________')
    for i in range(0,15,3):       
        print(hand_vec[i:i+3])




if __name__ == '__main__':
    main()