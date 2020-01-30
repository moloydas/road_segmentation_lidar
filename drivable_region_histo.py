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

for pts_itr in range(48,181,3):
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
            range_image[int(pt[4]),int(columnIdx),5] = dist
            if pt[4] < 5:
                poi.append(pt)

    for i in range(range_image.shape[0]):
        augmented_range_image[i,:,0] = utility.simple_augment_holes(range_image[i,:,0].copy(),.1)
        augmented_range_image[i,:,1] = utility.simple_augment_holes(range_image[i,:,1].copy(),.1)
        augmented_range_image[i,:,2] = utility.simple_augment_holes(range_image[i,:,2].copy(),.1)
        augmented_range_image[i,:,5] = utility.simple_augment_holes(range_image[i,:,5].copy(),.1)
        # plt.plot(augmented_range_image[i,:,5])
        # plt.show()

    drivable_region = []

    # for i in range(augmented_range_image.shape[1]):
    #     for j in range(0,5):
    #         if augmented_range_image[j,i,0] == 0 and augmented_range_image[j,i,1] == 0 and augmented_range_image[j,i,2] == 0:
    #             continue
    #         dz = augmented_range_image[j,i,2] - augmented_range_image[j+1,i,2]
    #         dx = augmented_range_image[j,i,0] - augmented_range_image[j+1,i,0]
    #         dy = augmented_range_image[j,i,1] - augmented_range_image[j+1,i,1]

    #         if dx != 0 or dy != 0 or dz != 0:
    #             slope = dz/(np.sqrt( np.square(dx) + np.square(dy)))
    #             if abs(slope) < 0.18 and augmented_range_image[j+1,i,2] < -1.1:
    #                 drivable_region.append(augmented_range_image[j+1,i,:3])
    #             else:
    #                 break

    max_freq = []

    for i in range(0,5):
        hist,_ = np.histogram(augmented_range_image[i,:,2], bins=50,range=(-2,-1))
        max_freq.append((i,_[hist.argmax()]))

        #plot results
        # plt.plot(sm_x)
        # plt.plot(m)
        # plt.plot(infl_pts[:,0],infl_pts[:,1],'go')
        # plt.plot(np.array(ifl_img)[:,0],np.array(ifl_img)[:,1],'yo')
        # plt.show()

    model = LineModel()
    model.estimate(np.array(max_freq))

    model_robust, inliers = ransac(np.array(max_freq), LineModel, min_samples=2, residual_threshold=.01, max_trials=1000)
    outliers = inliers == False
    y = model_robust.predict_y(np.arange(0,5))
    max_val = y + .05
    min_val = y - .5
    # params = curve_fit(utility.exp_fit_fn,np.array(max_freq)[:,0],np.array(max_freq)[:,1])
    # [a,b] = params[0]

    # plt.plot(np.array(max_freq)[:,0],np.array(max_freq)[:,1],'o')
    # plt.plot(np.arange(0,5),y)
    # plt.plot(np.arange(0,5),max_val,'ro')
    # plt.plot(np.arange(0,5),min_val,'ro')
    # plt.show()

    for i in range(0,5):
        for j in range(augmented_range_image.shape[1]):
            if augmented_range_image[i, j, 2] < max_val[i] and augmented_range_image[i, j, 2] > min_val[i]:
                drivable_region.append(augmented_range_image[i, j, :3])

    print("exec time: " + str(time.time() - start_time))

    drivable_region = np.array(drivable_region)

    pc = np.array(poi)

    geom = []

    # visualize pointcloud
    # pts_pcd = o3d.geometry.PointCloud()
    # pts_pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
    # pts_pcd.paint_uniform_color(np.array([[0.0],[0.8],[0.9]], dtype=np.float64))
    # geom.append(pts_pcd)

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
