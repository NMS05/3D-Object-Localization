# This script can localize a single object w.r.to a camera frame (Intel RealSense Depth Camera) in cartesian co-ordinates (x,y,z)
# yoloV3 detections are used to detect and localize the object (say a person) in 2D RGB image frame...... in (x,y)
# Using Yolo detections the ROI is extracted from the depth image to calculate the distance between camera and object in meters
# (x,y,dist) ------> (theta,xi,dist) : Localizing the object in Spherical or Polar co-ordinate system. 
# Theta,Xi are calculated assuming a pinhole camera model - https://www.sciencedirect.com/topics/engineering/pinhole-camera-model
# (theta,xi,dist) ------> (cart_x,cart_y,cart_z)  : Convert Spherical co-ordinate system to Cartesian co-ordinate system

import math
import cv2
import numpy as np
import pyrealsense2 as rs
import sys
import darknet


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
colorizer = rs.colorizer()
align = rs.align(rs.stream.color)

# Start streaming
profile = pipeline.start(config)

# converting BB coordinates
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


# drawing bounding boxes in image from detections
def cvDrawBoxes(detections, img):

    if len(detections) == 0: # if no detections are available, return fixed image center location. Cannot be empty.
        return(img, 320, 240)

    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (255, 255, 255), 1)
        cv2.putText(img, detection[0].decode(
        ), (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 255, 255], 2)
    return(img, int(x), int(y))


# camera parameters obtained after calibration
cx = 641.67604
cy = 358.33227
fx = 921.01612
fy = 916.67117

# convert polar to cartesian co-ordinates
def convert_to_cartesian(x, y, dist):
    theta = math.radians(90) - math.atan((cx - x)/fx)
    xi = math.radians(90) - math.atan((cy - y)/fy)
    X = dist * math.sin(xi) * math.cos(theta)
    Y = dist * math.sin(xi) * math.sin(theta)
    Z = dist * math.cos(xi)

    print("polar (yaw,pitch,dist) values are...",
          (-theta + math.radians(90), -xi + math.radians(90), dist))
    print("cartesian (X,Y,Z)  values are ...", (X, Y, Z))

# Specify YOLO model
IP_configPath = "cfg/yolov3.cfg"
IP_weightPath = "yolov3.weights"
IP_metaPath = "cfg/coco.data"

darknet.set_gpu(1)

IP_netMain = darknet.load_net_custom(IP_configPath.encode(
    "ascii"), IP_weightPath.encode("ascii"), 0, 1)  # batch size = 1
IP_metaMain = darknet.load_meta(IP_metaPath.encode("ascii"))


# Main Loop
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        bgr_color_image = np.asanyarray(color_frame.get_data())
        rgb_color_image = cv2.cvtColor(bgr_color_image, cv2.COLOR_BGR2RGB)


        colorized_depth_image = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())

        darknet_image = darknet.make_image(darknet.network_width(IP_netMain), darknet.network_height(IP_netMain), 3)
        darknet.copy_image_from_bytes(darknet_image, rgb_color_image.tobytes())
        detections = darknet.detect_image(IP_netMain, IP_metaMain, darknet_image, thresh=0.8)
        bgr_color_image, x, y = cvDrawBoxes(detections, bgr_color_image)

        # extract a small patch of 60x60 pixels
        pt1 = (x-30, y-30)
        pt2 = (x+30, y+30)

        # Show patch
        colorized_depth_image = cv2.rectangle(colorized_depth_image, pt1, pt2, (255, 255, 255), 1)

        # Stack both images horizontally
        images = np.hstack((bgr_color_image, colorized_depth_image))

        # Crop depth data:
        depth = np.asanyarray(depth_frame.get_data())
        depth = depth[x-30:x+30, y-30:y+30].astype(float)

        # Get data scale from the device and convert to meters
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth = depth * depth_scale
        #dist = np.mean(np.asarray(depth))
        dist, _, _, _ = cv2.mean(depth)
        print("\n\n\nDetected {0} meters away...".format(dist))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1000)

        convert_to_cartesian(x, y, dist*100)

finally:
    # Stop streaming
    pipeline.stop()
