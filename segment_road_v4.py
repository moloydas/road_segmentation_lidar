import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from skimage.measure import LineModel, ransac
import utility
import time

theta = np.array([-15, -13, -11, -9, -7, -5, -4])

for pts_itr in range(1,160,5):
    pts = np.load(str(pts_itr)+'.npy')
    print("loaded: " + str(pts_itr) + ".npy")
    N_SCAN = 16
    Horizon_SCAN = 1800
    ang_res_x = 0.2
    ang_res_y = 2.0
    ang_bottom = 15.0+0.1

    range_image = np.zeros((N_SCAN,1800,7))
    augmented_range_image = np.zeros(range_image.shape)
    dist = []

    start_time = time.time()

    pc = pts[pts[:,4] < 6]
    aug_start = time.time()

    for pt in pc:
        horizonAngle = math.atan2(pt[1], pt[0]) * 180 / math.pi
        dist = np.sqrt(np.square(pt[0]) + np.square(pt[1]) + np.square(pt[2]))
        if(horizonAngle < 60 and horizonAngle > -60) and dist > 1 and dist < 40:
            columnIdx = round(-1*horizonAngle/ang_res_x) + 450
            range_image[int(pt[4]),int(columnIdx),:5] = pt
            range_image[int(pt[4]),int(columnIdx),6] = np.sqrt(np.square(pt[0]) + np.square(pt[1]))

    for i in range(6):
        augmented_range_image[i,:,0] = utility.simple_augment_holes(range_image[i,:,0].copy(),.1)
        augmented_range_image[i,:,1] = utility.simple_augment_holes(range_image[i,:,1].copy(),.1)
        augmented_range_image[i,:,2] = utility.simple_augment_holes(range_image[i,:,2].copy(),.1)
        augmented_range_image[i,:,6] = utility.simple_augment_holes(range_image[i,:,6].copy(),.1)

    ifl_img = []
    curb_pts = []
    left_curb_pts = []
    right_curb_pts = []

    print("aug time: " + str(time.time() - aug_start))

    for i in range(0,5):
        # hist,_ = np.histogram(augmented_range_image[i,:,6], bins=50,range=(2,40))
        # max_val = _[hist.argmax()] + 3

        hist,_ = np.histogram(augmented_range_image[i,:,2], bins=50,range=(-2,-1))
        max_val = _[hist.argmax()] + .2

        print("max_val: "+ str(max_val))

        sm_x = gaussian_filter1d(augmented_range_image[i,:,0], 2)
        sm_y = gaussian_filter1d(augmented_range_image[i,:,1], 2)
        dx = np.gradient(sm_x)
        dy = np.gradient(sm_y)
        m = np.arctan2(dy,dx)
        m_filt = signal.medfilt(m,11)

        poi_start = time.time()
        infl_pts = utility.poi_5(m_filt, augmented_range_image[i,:,:], max_val)
        # max_dist = np.max(augmented_range_image[i,augmented_range_image[i,:,5] > 0,6])

        # print("poi time: " + str(time.time() - poi_start))

        # left_curb = np.array([0,10,0])
        # right_curb = np.array([0,-10,0])
        left_pts = []
        right_pts = []

        min_dist = _[hist.argmax()] / np.tan(np.radians(theta[i]))

        for j in range(infl_pts.shape[0]):
            infl_idx = int(infl_pts[j,0])
            if abs(augmented_range_image[i,infl_idx,1]) <= 11 and abs(augmented_range_image[i,infl_idx,1]) >= 0.5 and abs(augmented_range_image[i,infl_idx,6] - min_dist) < 3:
                # if augmented_range_image[i,int(infl_pts[j,0]),6] < max_dist+0.5 and augmented_range_image[i,int(infl_pts[j,0]),6] > max_dist-2:
                plt.plot(infl_pts[j,0], augmented_range_image[i,int(infl_pts[j,0]),6], 'ro')
                curb_pts.append(augmented_range_image[i,infl_idx,:3])

                if augmented_range_image[i,int(infl_pts[j,0]),1] > 0:
                    left_pts.append(augmented_range_image[i,int(infl_pts[j,0]),:3])

                elif augmented_range_image[i,int(infl_pts[j,0]),1] < 0:
                    right_pts.append(augmented_range_image[i,int(infl_pts[j,0]),:3])

        # min_height_left = np.min(np.array(left_pts)[:,2])
        # min_height_right = np.min(np.array(right_pts)[:,2])

        if len(left_pts) > 0:
            min_height_left = np.min(np.array(left_pts)[:,2])
            print("min height left: " + str(min_height_left))
            for pt in left_pts:
                if pt[2] < min_height_left+0.13:
                    left_curb_pts.append(pt)

        if len(right_pts) > 0:
            min_height_right = np.min(np.array(right_pts)[:,2])
            print("min height right: " + str(min_height_right))
            for pt in right_pts:
                if pt[2] < min_height_right+0.13:
                    right_curb_pts.append(pt)

        #plot results
        # plt.plot(augmented_range_image[i,:,0])
        # plt.plot(augmented_range_image[i,:,1])
        # plt.plot(augmented_range_image[i,:,2])
        # plt.plot(augmented_range_image[i,:,6])
        # plt.plot(m_filt)
        # plt.plot(infl_pts[:,0],infl_pts[:,1],'go')
        # plt.plot(augmented_range_image[i+1,:,6] - augmented_range_image[i,:,6])
        # plt.show()

    if len(left_curb_pts) > 0:
        model_left, inliers = ransac( np.array(left_curb_pts)[:,:2], LineModel, min_samples=2, residual_threshold=.5, max_trials=1000)
        outliers = inliers == False
        x_left = np.arange(5,15,0.2)
        y_left = model_left.predict_y(x_left)
        z_left = -1.5

    if len(right_curb_pts) > 0:
        model_right, inliers = ransac( np.array(right_curb_pts)[:,:2], LineModel, min_samples=2, residual_threshold=1, max_trials=1000)
        outliers = inliers == False
        x_right = np.arange(5,15,0.2)
        y_right = model_right.predict_y(x_right)
        z_right = -1.5

    print("exec time: " + str(time.time() - start_time))

    geom = []

    # visualize pointcloud

    for pt in curb_pts:
        pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        pcd.paint_uniform_color(np.array([[1.0],[0.0],[0.0]], dtype=np.float64))
        pcd.translate(pt)
        geom.append(pcd)
    
    for pt in left_curb_pts:
        pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
        pcd.paint_uniform_color(np.array([[1.0],[1.0],[0.0]], dtype=np.float64))
        pcd.translate(pt)
        geom.append(pcd)
    
    for pt in right_curb_pts:
        pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
        pcd.paint_uniform_color(np.array([[1.0],[0.0],[1.0]], dtype=np.float64))
        pcd.translate(pt)
        geom.append(pcd)

    for i in range(y_left.shape[0]):
        pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        pcd.paint_uniform_color(np.array([[0.0],[0.5],[0.5]], dtype=np.float64))
        pcd.translate([x_left[i],y_left[i],z_left])
        geom.append(pcd)
    
    for i in range(y_right.shape[0]):
        pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
        pcd.paint_uniform_color(np.array([[0.0],[0.5],[0.5]], dtype=np.float64))
        pcd.translate([x_right[i],y_right[i],z_right])
        geom.append(pcd)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=np.array([0., 0., 0.]))
    geom.append(mesh_frame)

    pcd_test = o3d.geometry.PointCloud()
    pcd_test.points = o3d.utility.Vector3dVector(pc[:,:3])
    pcd_test.paint_uniform_color(np.array([[0.0],[0.8],[0.0]], dtype=np.float64))
    geom.append(pcd_test)

    o3d.visualization.draw_geometries(geom)