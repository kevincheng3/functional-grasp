import numpy as np

def joint_constraint():
    data = np.array([
        [-0.25, 0.25],
        [-0.25, 0.25],
        [-0.3, 0.2],

        [-3, -1],
        [-1, 1],
        [-3.14, 3.14],


        [0, 1.571],
        [0, 1.571],
        [0, 1.571],    
        [0, 1.571],
        [0, 2],
        [0, 1.571],


    ])
    return data.T

def contact_location():
    "simple contact"
    # data = np.array([
    #     [-0.015, 0.012, 0.03],
    #     [-0.02, -0.012, 0.03],
    #     [0, 1, 0],
    #     [0, -1, 0],
    # ])
    "complete contact"

    data = np.array([

        [0.033, 0.001, 0.16],
        [0.035, 0.008, 0.135],
        [0.032, 0.005, 0.11],
        [0.022, 0.001, 0.09],
        [-0.035, 0.0, 0.13],        
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
    ])

    return data.ravel()

def ini_value():
    data = np.array([
        # -0.0225, 0.064, 0.208, 0.307, 0, 0, 0, 1.02, 0.456, 0, 0.482, 0.877, 0.223, 0.147, -0.236
        0.01, -0.15, -0.1, -1.41, 0.251, 3.14, 0, 0, 0, 0, 0, 0
    ])
    return data

def main():
    # data = joint_constraint()

    contact_data = contact_location()
    constraint_data = joint_constraint()
    ini_data = ini_value()
    # constraint = np.load("./constraint_data.npy")
    # data1 = constraint[0]
    # data = data1[[0,1,2,3,4,5,8,9,10,11,15,25,26,27,28,29]]
    # print(data)
    np.save("functional grasp/data/wine_glass/all_contact_data.npy", contact_data)
    np.save("functional grasp/data/wine_glass/constraint_data.npy", constraint_data)
    np.save("functional grasp/data/wine_glass/ini.npy", ini_data)


if __name__ == "__main__":
    main()
