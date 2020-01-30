import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import utility
import time

for pts_itr in range(1,2):
    pts = np.load(str(pts_itr)+'.npy')
    print("loaded: " + str(pts_itr) + ".npy" )
    N_SCAN = 16
    Horizon_SCAN = 1800
    ang_res_x = 0.2
    ang_res_y = 2.0
    ang_bottom = 15.0+0.1

    range_image = np.zeros((N_SCAN,1800,5))
    augmented_range_image = np.zeros(range_image.shape)
    dist = []

    start_time = time.time()

    poi = []

    for pt in pts:
        horizonAngle = math.atan2(pt[1], pt[0]) * 180 / math.pi
        if(horizonAngle < 90 and horizonAngle > -90):
            columnIdx = round(-1*horizonAngle/ang_res_x) + 450
            range_image[int(pt[4]),int(columnIdx),:] = pt
            if pt[4] < 5:
                poi.append(pt)

    for i in range(range_image.shape[0]):
        augmented_range_image[i,:,0] = utility.simple_augment_holes(range_image[i,:,0].copy(),.1)
        augmented_range_image[i,:,1] = utility.simple_augment_holes(range_image[i,:,1].copy(),.1)
        augmented_range_image[i,:,2] = utility.simple_augment_holes(range_image[i,:,2].copy(),.1)

    ifl_img = []
    curb_pts = []

    for i in range(4,5):
        hist,_ = np.histogram(augmented_range_image[i,:,2], bins=10,range=(-2,-1))
        min_val = _[hist.argmax()]
        max_val = _[hist.argmax()] + _[1] - _[0]

        ifl_img[:] = []

        for j in range(augmented_range_image.shape[1]):
            if augmented_range_image[i, j, 2] < max_val and augmented_range_image[i, j, 2] > min_val:
                ifl_img.append((j,augmented_range_image[i, j, 0]))

        sm_x = gaussian_filter1d(augmented_range_image[i,:,0], 3)
        sm_y = gaussian_filter1d(augmented_range_image[i,:,1], 3)
        dx = np.gradient(sm_x)
        dy = np.gradient(sm_y)
        m = gaussian_filter1d(np.arctan2(dy,dx), 1)
        # infl_pts = utility.inflection_points_2(m, 0.000001)
        infl_pts = utility.poi(m, 0.000001)
        idx_avg = np.average(np.array(ifl_img)[:,0])

        for j in range(infl_pts.shape[0]-1):
            if infl_pts[j,0] <= idx_avg and infl_pts[j+1,0] >= idx_avg:
                plt.plot(infl_pts[j,0], augmented_range_image[i,int(infl_pts[j,0]),0], 'ro')
                plt.plot(infl_pts[j+1,0], augmented_range_image[i,int(infl_pts[j+1,0]),0], 'ro')
                curb_pts.append(augmented_range_image[i,int(infl_pts[j,0]),:3])
                curb_pts.append(augmented_range_image[i,int(infl_pts[j+1,0]),:3])

        #plot results
        plt.plot(augmented_range_image[i,:,2])
        # plt.plot(m, 'ro')
        # plt.plot(infl_pts[:,0],infl_pts[:,1],'go')
        plt.plot(np.array(ifl_img)[:,0],np.array(ifl_img)[:,1],'yo')
        plt.show()

    print("curb pts:" + str(curb_pts))
    print("exec time: " + str(time.time() - start_time))

    # for i in range(0,5):
    #     hist,_ = np.histogram(augmented_range_image[i,:,2], bins=100,range=(-2,-1))
    #     min_val = _[hist.argmax()]
    #     max_val = _[hist.argmax()] + _[1] - _[0]

    #     sm_scan_line = gaussian_filter1d(gaussian_filter1d(augmented_range_image[i,:,2], 5),3)
    #     infl_pts = utility.inflection_points(sm_scan_line)

    #     for j in range(infl_pts.shape[0]):
    #         if augmented_range_image[i, int(infl_pts[j,0]),2] < max_val:
    #             ifl_img.append(augmented_range_image[i, int(infl_pts[j,0]),:3])

        # plt.plot(sm_scan_line)
        # plt.plot(infl_pts[:,0],infl_pts[:,1],'go')
        # plt.show()

        # for j in range(augmented_range_image.shape[1]):
        #     if augmented_range_image[i,j] > min_val and augmented_range_image[i,j] < max_val:
        #         plt.plot(j,augmented_range_image[i,j],'ro')

    # curb_pts = np.array(curb_pts)

    # # for i in range(0,5):
    # #     sm_scan_line = gaussian_filter1d(augmented_range_image[i,:], 5)
    # #     infl_pts = utility.inflection_points(sm_scan_line)
    # #     plt.plot(augmented_range_image[i,:])
    # #     # plt.plot(np.gradient(sm_scan_line)*10)
    # #     # plt.plot(infl_pts[:,0],infl_pts[:,1],'ro')
    # # plt.show()

    pc = np.array(poi)

    geom = []

    # visualize pointcloud
    for pt in curb_pts:
        pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
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
