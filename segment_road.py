import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter1d
import utility
import time

pts = np.load('velo_gps_1536236079659176067.npy')

N_SCAN = 16
Horizon_SCAN = 1800
ang_res_x = 0.2
ang_res_y = 2.0
ang_bottom = 15.0+0.1

range_image = np.zeros((N_SCAN,1800,5))
augmented_range_image = np.zeros(range_image.shape)
dist = []

for pt in pts:
    horizonAngle = math.atan2(pt[1], pt[0]) * 180 / math.pi
    if(horizonAngle < 90 and horizonAngle > -90):
        columnIdx = round(-1*horizonAngle/ang_res_x) + 450
        # range_image[int(pt[4]),int(columnIdx)] = np.sum(np.square(pt[:3]))
        range_image[int(pt[4]),int(columnIdx),:] = pt

for i in range(range_image.shape[0]):
    augmented_range_image[i,:,0] = utility.simple_augment_holes(range_image[i,:,0].copy(),.1)
    augmented_range_image[i,:,1] = utility.simple_augment_holes(range_image[i,:,1].copy(),.1)
    augmented_range_image[i,:,2] = utility.simple_augment_holes(range_image[i,:,2].copy(),.1)

ifl_img = []

for i in range(0,5):
    plt.plot(augmented_range_image[i,:,1])
    plt.show()

for i in range(0,5):
    hist,_ = np.histogram(augmented_range_image[i,:,2], bins=100,range=(-2,-1))
    min_val = _[hist.argmax()]
    max_val = _[hist.argmax()] + _[1] - _[0]

    sm_scan_line = gaussian_filter1d(gaussian_filter1d(augmented_range_image[i,:,2], 5),3)
    infl_pts = utility.inflection_points(sm_scan_line)

    for j in range(infl_pts.shape[0]):
        if augmented_range_image[i, int(infl_pts[j,0]),2] < max_val:
            ifl_img.append(augmented_range_image[i, int(infl_pts[j,0]),:3])

    # plt.plot(sm_scan_line)
    # plt.plot(infl_pts[:,0],infl_pts[:,1],'go')
    # plt.show()

    # for j in range(augmented_range_image.shape[1]):
    #     if augmented_range_image[i,j] > min_val and augmented_range_image[i,j] < max_val:
    #         plt.plot(j,augmented_range_image[i,j],'ro')

ifl_img = np.array(ifl_img)

# for i in range(0,5):
#     sm_scan_line = gaussian_filter1d(augmented_range_image[i,:], 5)
#     infl_pts = utility.inflection_points(sm_scan_line)
#     plt.plot(augmented_range_image[i,:])
#     # plt.plot(np.gradient(sm_scan_line)*10)
#     # plt.plot(infl_pts[:,0],infl_pts[:,1],'ro')
# plt.show()

pc = pts[pts[:,4] < 5]

geom = []

# visualize pointcloud

for pt in ifl_img:
    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    pcd.paint_uniform_color(np.array([[1.0],[0.0],[0.0]], dtype=np.float64))
    pcd.translate(pt)
    geom.append(pcd)

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=np.array([0., 0., 0.]))
geom.append(mesh_frame)

pcd_test = o3d.geometry.PointCloud()
pcd_test.points = o3d.utility.Vector3dVector(pc[:,:3])
pcd_test.paint_uniform_color(np.array([[0.0],[0.8],[0.0]], dtype=np.float64))
geom.append(pcd_test)

o3d.visualization.draw_geometries(geom)
