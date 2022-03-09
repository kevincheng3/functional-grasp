import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
# [-0.02, 0.012, 0.03],
# [0.0, 0.012, 0.03],
# [0.02, 0.012, 0.03],
# [0.04, 0.012, 0.03],
# [-0.02, -0.012, 0.03],
# [0, 1, 0],
# [0, 1, 0],
# [0, 1, 0],
# [0, 1, 0],
# [0, -1, 0],


centroid = np.array([-1.63407611e-03, 5.41759906e-06, 1.81941864e-02])
B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])


index_p = np.array([-0.02, 0.012, 0.03])
index_R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
index_p = np.cross((index_p - centroid), np.identity(index_p.shape[0]) * -1)
# print(index_p.dot(index_R))
index_AdG = np.block([
    [index_R, np.zeros((3,3))],
    [index_p.dot(index_R), index_R]
])
index_G = index_AdG.dot(B)
"----------------------------------------------------------------"
middle_p = np.array([0.0, 0.012, 0.03])
middle_R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
middle_p = np.cross((middle_p - centroid), np.identity(middle_p.shape[0]) * -1)

middle_AdG = np.block([
    [middle_R, np.zeros((3,3))],
    [middle_p.dot(middle_R), middle_R]
])
middle_G = middle_AdG.dot(B)
"----------------------------------------------------------------"
ring_p = np.array([0.02, 0.012, 0.03])
ring_R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
ring_p = np.cross((ring_p - centroid), np.identity(ring_p.shape[0]) * -1)

ring_AdG = np.block([
    [ring_R, np.zeros((3,3))],
    [ring_p.dot(ring_R), ring_R]
])
ring_G = ring_AdG.dot(B)
"----------------------------------------------------------------"
little_p = np.array([0.02, 0.012, 0.03])
little_R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
little_p = np.cross((little_p - centroid), np.identity(little_p.shape[0]) * -1)

little_AdG = np.block([
    [little_R, np.zeros((3,3))],
    [little_p.dot(little_R), little_R]
])
little_G = little_AdG.dot(B)
"----------------------------------------------------------------"
thumb_p = np.array([-0.02, -0.012, 0.03])
thumb_R = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
thumb_p = np.cross((thumb_p - centroid), np.identity(thumb_p.shape[0]) * -1)

thumb_AdG = np.block([
    [thumb_R, np.zeros((3,3))],
    [thumb_p.dot(thumb_R), thumb_R]
])
thumb_G = thumb_AdG.dot(B)

G = np.concatenate((index_G, middle_G, ring_G, little_G, thumb_G), axis=1)

pts = G.T


# print(G.T)
hull = ConvexHull(pts)

# print(hull.simplices)

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0
# pts = np.array([[-1, 0, -0.1], [0, -1, 0.1], [1, 0,- 0.2], [0, 1, 0.2]
#                  ])
print(in_hull(np.array([0, 0, 0, 0, 0, 0]), pts))


# from scipy.linalg import null_space


# def com(n: int, r: int)-> list:
#     if n == r:
#         return [[i for i in range(1,n+1)]]
#     if r == 1:
#         return [[i] for i in range(1,n+1)]
#     ans = []
#     # 先把n拿出来放进组合，然后从剩下的n-1个数里选出r-1个数
#     for each_list in com(n-1,r-1):
#         ans.append([n]+each_list)
#     # 直接从剩下的n-1个数里，选出r个
#     for each_list in com(n-1,r):
#         ans.append(each_list)
#     return ans
# index = np.array(com(15,5)) - 1


# determination = True

# for i in range(2000, index.shape[0]):
#     A = np.array([G.T[:][index[i][0]], G.T[:][index[i][1]], G.T[:][index[i][2]], G.T[:][index[i][3]], G.T[:][index[i][4]]])
#     ns = null_space(A)
#     result = ns.T[:][0].T.dot(G)
#     # print(ns.shape)
#     if np.any(result < -1e-18) == False:
#         print(i)
#         determination = False
#         break
#     elif ns.shape[1] ==2:
#         result = ns.T[:][1].T.dot(G)
#         print("   ", i)
#         if np.any(result < -1e-18) == False:
#             print(i)
#             determination = False
#             break

# print(determination)
