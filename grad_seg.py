# ros_segment_road.py
import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
import utility
import math
from scipy.ndimage import gaussian_filter1d
import time

N_SCAN = 16
Horizon_SCAN = 1800
ang_res_x = 0.2
ang_res_y = 2.0
ang_bottom = 15.0+0.1

pub_poi = rospy.Publisher("poi", PointCloud2,queue_size=1)

loi = 6

curb_pts = []
ifl_img = []

def read_pc(msg):
    start_time = time.time()

    pts_full = utility.pointcloud2_to_array(msg)
    pts = pts_full[pts_full[:,4] < loi]
    range_image = np.zeros((N_SCAN,1800,5))
    augmented_range_image = np.zeros(range_image.shape)

    for pt in pts:
        horizonAngle = math.atan2(pt[1], pt[0]) * 180 / math.pi
        if(horizonAngle < 90 and horizonAngle > -90):
            columnIdx = round(-1*horizonAngle/ang_res_x) + 450
            range_image[int(pt[4]),int(columnIdx),:] = pt

    for i in range(loi):
        augmented_range_image[i,:,0] = utility.simple_augment_holes(range_image[i,:,0].copy(),.1)
        augmented_range_image[i,:,1] = utility.simple_augment_holes(range_image[i,:,1].copy(),.1)
        augmented_range_image[i,:,2] = utility.simple_augment_holes(range_image[i,:,2].copy(),.1)

    curb_pts[:] = []

    for i in range(0,5):
        # hist,_ = np.histogram(augmented_range_image[i,:,2], bins=100,range=(-2,-1))
        # min_val = _[hist.argmax()]
        # max_val = _[hist.argmax()] + _[1] - _[0]

        # ifl_img[:] = []

        # for j in range(augmented_range_image.shape[1]):
        #     if augmented_range_image[i, j, 2] < max_val and augmented_range_image[i, j, 2] > min_val:
        #         ifl_img.append((j,augmented_range_image[i, j, 0]))

        sm_x = gaussian_filter1d(augmented_range_image[i,:,0], 5)
        sm_y = gaussian_filter1d(augmented_range_image[i,:,1], 5)
        dx = np.gradient(sm_x)
        dy = np.gradient(sm_y)
        m = gaussian_filter1d(np.arctan2(dy,dx), 9)
        infl_pts = utility.inflection_points_2(m, 0.000001)

        for j in range(infl_pts.shape[0]-1):
            curb_pts.append(augmented_range_image[i,int(infl_pts[j,0]),:])
            curb_pts.append(augmented_range_image[i,int(infl_pts[j+1,0]),:])

    poi_msg = utility.array_to_pointcloud2(np.array(curb_pts), msg.header)

    pub_poi.publish(poi_msg)

    print("time taken: " + str(time.time() - start_time))

if __name__ == '__main__':
    rospy.Subscriber("/velodyne_points", PointCloud2, read_pc)
    rospy.init_node('Segment_Road', anonymous=True)
    rospy.spin()
