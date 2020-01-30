# ros_segment_road.py
import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
import utility
import math
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from skimage.measure import LineModel, ransac
import time

N_SCAN = 16
Horizon_SCAN = 1800
ang_res_x = 0.2
ang_res_y = 2.0
ang_bottom = 15.0+0.1

pub_poi = rospy.Publisher("poi", PointCloud2,queue_size=1)
pub_road_boundary = rospy.Publisher("road_boundary", PointCloud2,queue_size=1)
pub_pts = rospy.Publisher("velodyne_points_less", PointCloud2,queue_size=1)

loi = 6

curb_pts = []
ifl_img = []

def read_pc(msg):
    start_time = time.time()

    pts_full = utility.pointcloud2_to_array(msg)
    pts = pts_full[pts_full[:,4] < loi]
    range_image = np.zeros((N_SCAN,1800,7))
    augmented_range_image = np.zeros(range_image.shape)

    for pt in pts:
        horizonAngle = math.atan2(pt[1], pt[0]) * 180 / math.pi
        dist = np.sqrt(np.square(pt[0]) + np.square(pt[1]) + np.square(pt[2]))
        if(horizonAngle < 90 and horizonAngle > -90) and dist > 1 and dist < 40:
            columnIdx = round(-1*horizonAngle/ang_res_x) + 450
            range_image[int(pt[4]),int(columnIdx),:5] = pt
            range_image[int(pt[4]),int(columnIdx),6] = np.sqrt(np.square(pt[0]) + np.square(pt[1]))

    for i in range(6):
        augmented_range_image[i,:,0] = utility.simple_augment_holes(range_image[i,:,0].copy(),.1)
        augmented_range_image[i,:,1] = utility.simple_augment_holes(range_image[i,:,1].copy(),.1)
        augmented_range_image[i,:,2] = utility.simple_augment_holes(range_image[i,:,2].copy(),.1)
        augmented_range_image[i,:,6] = utility.simple_augment_holes(range_image[i,:,6].copy(),.1)

    curb_pts = []
    left_curb_pts = []
    right_curb_pts = []


    for i in range(0,5):
        hist,_ = np.histogram(augmented_range_image[i,:,6], bins=50,range=(2,40))
        max_val = _[hist.argmax()] + 2

        sm_x = gaussian_filter1d(augmented_range_image[i,:,0], 2)
        sm_y = gaussian_filter1d(augmented_range_image[i,:,1], 2)
        dx = np.gradient(sm_x)
        dy = np.gradient(sm_y)
        m = np.arctan2(dy,dx)
        m_filt = signal.medfilt(m,11)

        poi_start = time.time()
        infl_pts = utility.poi_5(m_filt, augmented_range_image[i,:,:], max_val)
        max_dist = np.max(augmented_range_image[i,augmented_range_image[i,:,5] > 0,6])

        print("poi time: " + str(time.time() - poi_start))

        # left_curb = np.array([0,10,0])
        # right_curb = np.array([0,-10,0])
        left_pts = []
        right_pts = []

        for j in range(infl_pts.shape[0]):
            infl_idx = int(infl_pts[j,0])
            if abs(augmented_range_image[i,infl_idx,1]) <= 7 and augmented_range_image[i,infl_idx,2] < -1.1:
                if augmented_range_image[i,int(infl_pts[j,0]),6] < max_dist+0.5 and augmented_range_image[i,int(infl_pts[j,0]),6] > max_dist-2:
                    # plt.plot(infl_pts[j,0], augmented_range_image[i,int(infl_pts[j,0]),6], 'ro')
                    curb_pts.append(augmented_range_image[i,infl_idx,:3])

                    if augmented_range_image[i,int(infl_pts[j,0]),1] > 0:
                        left_pts.append(augmented_range_image[i,int(infl_pts[j,0]),:3])

                    elif augmented_range_image[i,int(infl_pts[j,0]),1] < 0:
                        right_pts.append(augmented_range_image[i,int(infl_pts[j,0]),:3])

        if len(left_pts) > 0:
            min_height_left = np.min(np.array(left_pts)[:,2])
            for pt in left_pts:
                if pt[2] < min_height_left+0.1:
                    left_curb_pts.append(pt)

        if len(right_pts) > 0:
            min_height_right = np.min(np.array(right_pts)[:,2])
            for pt in right_pts:
                if pt[2] < min_height_right+0.1:
                    right_curb_pts.append(pt)

    model_left, inliers = ransac( np.array(left_curb_pts)[:,:2], LineModel, min_samples=2, residual_threshold=.5, max_trials=1000)
    outliers = inliers == False
    x_left = np.arange(5,15,0.2)
    y_left = model_left.predict_y(x_left)
    z_left = -1.5

    model_right, inliers = ransac( np.array(right_curb_pts)[:,:2], LineModel, min_samples=2, residual_threshold=1, max_trials=1000)
    outliers = inliers == False
    x_right = np.arange(5,15,0.2)
    y_right = model_right.predict_y(x_right)
    z_right = -1.5

    road_boundary = []

    for i in range(0,y_left.shape[0]):
        road_boundary.append([x_left[i],y_left[i],z_left])
        road_boundary.append([x_right[i],y_right[i],z_right])

    poi_msg = utility.array_to_pointcloud2_without_intensity(np.array(curb_pts), msg.header)
    pub_poi.publish(poi_msg)
    pub_pts.publish(utility.array_to_pointcloud2(np.array(pts), msg.header))
    pub_road_boundary.publish(utility.array_to_pointcloud2_without_intensity(np.array(road_boundary), msg.header))

    print("time taken: " + str(time.time() - start_time))

if __name__ == '__main__':
    rospy.Subscriber("/velodyne_points", PointCloud2, read_pc)
    rospy.init_node('Segment_Road', anonymous=True)
    rospy.spin()
