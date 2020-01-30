# filter pointcloud

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn import linear_model
from skimage.measure import LineModel, ransac
import utility
import time

for pts_itr in range(1,181,3):
    pts = np.load(str(pts_itr)+'.npy')
    print("loaded: " + str(pts_itr) + ".npy" )
    N_SCAN = 16
    Horizon_SCAN = 1800
    ang_res_x = 0.2
    ang_res_y = 2.0
    ang_bottom = 15.0+0.1

    range_image = np.zeros((N_SCAN,1800,6))
    augmented_range_image = np.zeros(range_image.shape)

    start_time = time.time()

    poi = []

    for pt in pts:
        horizonAngle = math.atan2(pt[1], pt[0]) * 180 / math.pi
        dist = np.sqrt(np.square(pt[0]) + np.square(pt[1]) + np.square(pt[2]) )
        if(horizonAngle < 90 and horizonAngle > -90 and dist > 1):
            columnIdx = round(-1*horizonAngle/ang_res_x) + 450
            range_image[int(pt[4]),int(columnIdx),:5] = pt
            range_image[int(pt[4]),int(columnIdx),5] = np.sqrt(np.square(pt[0]) + np.square(pt[1]))
            if pt[4] < 5:
                poi.append(pt)

    for i in range(range_image.shape[0]):
        augmented_range_image[i,:,0] = utility.simple_augment_holes(range_image[i,:,0].copy(),.1)
        augmented_range_image[i,:,1] = utility.simple_augment_holes(range_image[i,:,1].copy(),.1)
        augmented_range_image[i,:,2] = utility.simple_augment_holes(range_image[i,:,2].copy(),.1)
        augmented_range_image[i,:,5] = utility.simple_augment_holes(range_image[i,:,5].copy(),.1)

    drivable_region = []

    window = 20

    for i in range(0,5):
        for j in range(augmented_range_image.shape[1] - window):
            if np.std(augmented_range_image[i,j:j+window,5]) < .5:
                drivable_region.append(augmented_range_image[i,j,:])
        
        # plt.plot(augmented_range_image[i,:,0])
        # plt.plot(np.array(drivable_region)[:,0],np.array(drivable_region)[:,1])
        # plt.show()

    print("exec time: " + str(time.time() - start_time))

    drivable_region = np.array(drivable_region)

    pc = np.array(poi)

    geom = []

    # visualize pointcloud
    pts_pcd = o3d.geometry.PointCloud()
    pts_pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
    pts_pcd.paint_uniform_color(np.array([[0.0],[0.8],[0.9]], dtype=np.float64))
    geom.append(pts_pcd)

    # for pt in drivable_region:
    #     pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    #     pcd.paint_uniform_color(np.array([[0.8],[0.8],[0.0]], dtype=np.float64))
    #     pcd.translate(pt)
    #     geom.append(pcd)

    drivable_pcd = o3d.geometry.PointCloud()
    drivable_pcd.points = o3d.utility.Vector3dVector(drivable_region[:,:3])
    drivable_pcd.paint_uniform_color(np.array([[0.0],[1.0],[0.0]], dtype=np.float64))
    geom.append(drivable_pcd)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=np.array([0., 0., 0.]))
    geom.append(mesh_frame)

    o3d.visualization.draw_geometries(geom)
