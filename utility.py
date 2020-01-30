# utility functions

import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Point32
import sensor_msgs.point_cloud2 as pc2
from scipy.ndimage import gaussian_filter1d

def augment_holes(arr,thres):
    val = 0
    val_2 = 0
    for i in range(arr.shape[0]):
        if arr[i] == 0 and val != 0:
            for j in range(i,arr.shape[0]):
                if arr[j] != 0 and abs(arr[j]) > thres:
                    val_2 = arr[j]
                    arr[i] = (val + val_2)/2
                    # val = arr[i]
                    break
        elif arr[i] != 0 and abs(arr[i]) > thres:
            val = arr[i]
    return arr

def simple_augment_holes(arr,thres):
    val = 0
    last_non_zero_idx = 0
    for i in range(arr.shape[0]):
        if arr[i] == 0 and val != 0:
            arr[i] = val
        elif arr[i] != 0 and abs(arr[i]) > thres:
            val = arr[i]
            last_non_zero_idx = i
    arr[(last_non_zero_idx+1):] = 0
    return arr

def inflection_points(arr):
    pts = []
    d = np.gradient(arr)
    for i in range(0,d.shape[0]):
        if d[i-1] * d[i] < 0:
            pts.append((i-1,arr[i-1]))
    return np.array(pts)

def inflection_points_2(arr, thres):
    pts = []
    d = np.gradient(arr)
    for i in range(0,d.shape[0]):
        if d[i-1] * d[i] < 0 and abs(d[i-1] * d[i]) > thres:
            pts.append((i-1,arr[i-1]))
    return np.array(pts)

def poi(arr, thres):
    pts = []
    d = np.gradient(arr)
    window = 30
    for i in range(1,d.shape[0]-window):
        group_avg = np.sum(arr[i:i+window])/window
        for j in range(i,i+window):
            if d[j-1] * d[j] < 0 and abs(d[j-1] * d[j]) > thres and abs(group_avg - arr[j-1]) > 0.35:
            # if d[j-1] * d[j] < 0 and abs(d[j-1] * d[j]) > thres:
                pts.append((j-1,arr[j-1]))
    return np.array(pts)

def poi_2(arr, thres):
    pts = []
    d = np.gradient(arr)
    dd = gaussian_filter1d(np.gradient(d),2)
    window = 30
    for i in range(1,dd.shape[0]):
        if abs(dd[i]) > 0.02:
            pts.append((i-1,arr[i-1]))
    return np.array(pts)

def poi_3(arr, thres):
    pts = []
    window = 10
    for i in range(0,arr.shape[0]-window):
        a = gaussian_filter1d(arr[i:i+window],1)
        d = np.gradient(a)
        for j in range(0, window-1):
            if abs(d[j]) >= 0.1 and abs(d[j]) <= 0.4:
                pts.append((i+j,arr[i+j]))
    return np.array(pts)

def poi_4(arr, thres):
    pts = []
    window = 15
    for i in range(0,arr.shape[0]-window):
        a = gaussian_filter1d(arr[i:i+window],7)
        if np.std(a) > 0.1:
            continue
        d = np.gradient(a)
        dd = np.gradient(d)
        for j in range(0, window-1,7):
            if abs(d[j]) >= 0.01 and abs(d[j]) <= 0.4 and abs(dd[j-1]) < 0.001:
                pts.append((i+j,arr[i+j]))
    return np.array(pts)

def poi_5(arr, data, max_val):
    pts = []
    window = 15
    sm_arr = gaussian_filter1d(arr,9)
    d = np.gradient(sm_arr)
    dd = np.gradient(d)
    for i in range(0,arr.shape[0]-window):
        a = sm_arr[i:i+window]
        if np.sum(data[i:i+window,5] > 0) == 0:
            # if i > 540 and i < 620:
            #     print(str(i) + " " + str(np.std(np.gradient(data[i:i+window,6]))))
            if np.std(np.gradient(data[i:i+window,6])) < 0.06:
                # continue

                for j in range(0, window-1):
                    if abs(d[i+j]) >= 0.01 and abs(d[i+j]) <= 0.4 and abs(dd[i+j-1]) < 0.0015 and data[i+j,2] < max_val:
                        pts.append((i+j,arr[i+j]))
                        data[i+j,5] = 1
                        break

    return np.array(pts)

def line_fit_fn(x,m,b):
    return m*x + b

def exp_fit_fn(x,a,b):
    return a*np.exp(b*x)

def pointcloud2_to_array(msg):
    pts_list = []
    for pts in pc2.read_points(msg, skip_nans=True):
        pts_list.append(pts)
    return np.array(pts_list)

def array_to_pointcloud2(arr, header):
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = arr.astype(dtype).tobytes()

    fields = [PointField( name=n, offset=i*itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyzil')]

    return PointCloud2(
    header=header,
    height=1,
    width=arr.shape[0],
    is_dense=False,
    is_bigendian=False,
    fields=fields,
    point_step=(itemsize * 5),
    row_step=(itemsize * 5 * arr.shape[0]),
    data=data
    )

def array_to_pointcloud2_without_intensity(arr, header):
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = arr.astype(dtype).tobytes()

    fields = [PointField( name=n, offset=i*itemsize, datatype=ros_dtype, count=1) for i, n in enumerate('xyz')]

    return PointCloud2(
    header=header,
    height=1,
    width=arr.shape[0],
    is_dense=False,
    is_bigendian=False,
    fields=fields,
    point_step=(itemsize * 3),
    row_step=(itemsize * 3 * arr.shape[0]),
    data=data
    )