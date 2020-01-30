import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import utility
import time

for pts_itr in range(1,180,10):
    pts = np.load(str(pts_itr)+'.npy')
    print("loaded: " + str(pts_itr) + ".npy" )
    N_SCAN = 16
    Horizon_SCAN = 1800
    ang_res_x = 0.2
    ang_res_y = 2.0
    ang_bottom = 15.0+0.1

    range_image = np.zeros((N_SCAN,1800,6))
    augmented_range_image = np.zeros(range_image.shape)
    dist = []

    start_time = time.time()

    pc = pts[pts[:,4] < 6]
    aug_start = time.time()
    for pt in pc:
        horizonAngle = math.atan2(pt[1], pt[0]) * 180 / math.pi
        dist = np.sqrt(np.square(pt[0]) + np.square(pt[1]) + np.square(pt[2]))
        if(horizonAngle < 90 and horizonAngle > -90) and dist > 1:
            columnIdx = round(-1*horizonAngle/ang_res_x) + 450
            range_image[int(pt[4]),int(columnIdx),:5] = pt

    for i in range(5):
        augmented_range_image[i,:,0] = utility.simple_augment_holes(range_image[i,:,0].copy(),.1)
        augmented_range_image[i,:,1] = utility.simple_augment_holes(range_image[i,:,1].copy(),.1)
        augmented_range_image[i,:,2] = utility.simple_augment_holes(range_image[i,:,2].copy(),.1)

    ifl_img = []
    curb_pts = []
    print("aug time: " + str(time.time() - aug_start))
    for i in range(0,5):
        # hist,_ = np.histogram(augmented_range_image[i,:,2], bins=10,range=(-2,-1))
        # min_val = _[hist.argmax()]
        # max_val = _[hist.argmax()] + _[1] - _[0]

        # ifl_img[:] = []

        # for j in range(augmented_range_image.shape[1]):
        #     if augmented_range_image[i, j, 2] < max_val and augmented_range_image[i, j, 2] > min_val:
        #         ifl_img.append((j,augmented_range_image[i, j, 0]))

        sm_x = gaussian_filter1d(augmented_range_image[i,:,0], 2)
        sm_y = gaussian_filter1d(augmented_range_image[i,:,1], 2)
        dx = np.gradient(sm_x)
        dy = np.gradient(sm_y)
        # m = gaussian_filter1d(np.arctan2(dy,dx), 1)
        m = np.arctan2(dy,dx)
        m_filt = signal.medfilt(m,15)

        poi_start = time.time()
        infl_pts = utility.poi_5(m_filt, augmented_range_image[i,:,:])
        print("poi time: " + str(time.time() - poi_start))

        for j in range(infl_pts.shape[0]):
            if augmented_range_image[i,int(infl_pts[j,0]),1] <= 10 and augmented_range_image[i,int(infl_pts[j,0]),1] >= -10 and augmented_range_image[i,int(infl_pts[j,0]),2] < -1.1:
                plt.plot(infl_pts[j,0], augmented_range_image[i,int(infl_pts[j,0]),0], 'ro')
                curb_pts.append(augmented_range_image[i,int(infl_pts[j,0]),:3])

        #plot results
        plt.plot(augmented_range_image[i,:,0])
        # plt.plot(augmented_range_image[i,:,1])
        # plt.plot(augmented_range_image[i,:,2])
        plt.plot(m_filt)
        plt.plot(infl_pts[:,0],infl_pts[:,1],'go')
        plt.show()
# 326
        # for k in range(320,m.shape[0]):
        #     print(k)
        #     window = 50
        #     pts = []
        #     a = gaussian_filter1d(m_filt[k:k+window],10)
        #     d = np.gradient(a)
        #     for j in range(0, window-1):
        #         if abs(d[j]) >= 0.01 and abs(d[j]) <= 0.4:
        #             pts.append((j,m[k+j]))
        #     plt.plot(d)
        #     plt.plot(gaussian_filter1d(m_filt[k:k+window],1))
        #     plt.plot(m[k:k+window])
        #     if len(pts) > 0:
        #         plt.plot(np.array(pts)[:,0],np.array(pts)[:,1],'yo')
        #     plt.show()

    # print("curb pts:" + str(curb_pts))
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

    geom = []

    # visualize pointcloud
    for pt in curb_pts:
        pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
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