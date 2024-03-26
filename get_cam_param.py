#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CameraInfo

def camera_info_callback(msg):
    fx = msg.K[0]
    fy = msg.K[4]
    cx = msg.K[2]
    cy = msg.K[5]
    print("fx:", fx)
    print("fy:", fy)
    print("cx:", cx)
    print("cy:", cy)

rospy.init_node('camera_info_listener', anonymous=True)
rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, camera_info_callback)
rospy.spin()

