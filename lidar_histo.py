# ros_segment_road.py

import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
import utility
import math
from scipy.ndimage import gaussian_filter1d
from skimage.measure import LineModel, ransac
import time

N_SCAN = 16
Horizon_SCAN = 1800
ang_res_x = 0.2
ang_res_y = 2.0
ang_bottom = 15.0+0.1

pub_road = rospy.Publisher("road", PointCloud2,queue_size=1)
pub_pts = rospy.Publisher("velodyne_points_less", PointCloud2,queue_size=1)

loi = 6

road_pts = []

def read_pc(msg):
    start_time = time.time()

    pts_full = utility.pointcloud2_to_array(msg)
    pts = pts_full[pts_full[:,4] < loi]
    range_image = np.zeros((N_SCAN,1800,loi))
    augmented_range_image = np.zeros(range_image.shape)

    for pt in pts:
        horizonAngle = math.atan2(pt[1], pt[0]) * 180 / math.pi
        dist = np.sqrt(np.square(pt[0]) + np.square(pt[1]) + np.square(pt[2]) )
        if(horizonAngle < 90 and horizonAngle > -90):
            columnIdx = round(-1*horizonAngle/ang_res_x) + 450
            range_image[int(pt[4]),int(columnIdx),:5] = pt
            range_image[int(pt[4]),int(columnIdx),5] = dist


    for i in range(loi):
        augmented_range_image[i,:,0] = utility.simple_augment_holes(range_image[i,:,0].copy(),.1)
        augmented_range_image[i,:,1] = utility.simple_augment_holes(range_image[i,:,1].copy(),.1)
        augmented_range_image[i,:,2] = utility.simple_augment_holes(range_image[i,:,2].copy(),.1)

    road_pts[:] = []

    max_freq = []

    for i in range(0,loi):
        hist,_ = np.histogram(augmented_range_image[i,:,2], bins=50,range=(-2,-1))
        max_freq.append((i,_[hist.argmax()]))

    model = LineModel()
    model.estimate(np.array(max_freq))

    model_robust, inliers = ransac(np.array(max_freq), LineModel, min_samples=2, residual_threshold=.01, max_trials=1000)
    outliers = inliers == False
    y = model_robust.predict_y(np.arange(0,5))
    max_val = y + .2
    min_val = y - .8

    for i in range(0,5):
        for j in range(augmented_range_image.shape[1]):
            if augmented_range_image[i, j, 2] < max_val[i] and augmented_range_image[i, j, 2] > min_val[i]:
                road_pts.append(augmented_range_image[i, j, :5])

    road_msg = utility.array_to_pointcloud2(np.array(road_pts), msg.header)
    velodyne_less_msg = utility.array_to_pointcloud2(pts, msg.header)

    pub_road.publish(road_msg)
    pub_pts.publish(velodyne_less_msg)
    print("time taken: " + str(time.time() - start_time))

if __name__ == '__main__':
    rospy.Subscriber("/velodyne_points", PointCloud2, read_pc)
    rospy.init_node('Segment_Road', anonymous=True)
    rospy.spin()
