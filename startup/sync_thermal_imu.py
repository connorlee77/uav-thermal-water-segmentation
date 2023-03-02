import time
import cv2
from skimage import exposure
import numpy as np
import quaternion
import argparse

# ROS integration
import rospy
import message_filters
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge

import config.flir_boson as flir_boson
from utils.utils import raw16bit_to_32FC_autoScale
from horizon.horizon import project_points, estimate_horizon_mask, draw_overlay_and_pts


class ThermalImuSync:
    
    def __init__(self, rotate_180=False):
        
        self.rotate_180 = rotate_180

        # Subscriber topics
        self.thermal_raw_topic = '/boson/thermal/image_raw'
        self.imu_topic = '/imu/imu'
        
        # image width, height
        self.H, self.W = (512, 640)

        # CV bridge
        self.bridge = CvBridge()

        # Compute constants and start ros
        self.compute_constants()
        self.start_ros()

    def compute_constants(self):    
        #  [ x, y, z, qw, qx, qy, qz ] Transform to "aircraft axis definition" - x is fwd, Y right, Z down
        imu2adk_tf2 = np.array([0.013, 0.000, 0.027, 0, 1.0, 0.0, 0.0])
        self.imu2adk_quat = quaternion.from_float_array(imu2adk_tf2[3:])

        self.newcameramtx, roi = cv2.getOptimalNewCameraMatrix(flir_boson.I, flir_boson.D, (self.W, self.H), 0, (self.W, self.H))
        self.new_P = np.hstack([self.newcameramtx, np.zeros((3,1))])
        self.clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(16, 16))

    def start_ros(self):

        # Create the node
        rospy.init_node('thermal_imu_sync', anonymous=True)

        # Publisher topics
        thermal_eq_out = self.thermal_raw_topic.replace('raw', 'rect_histeq')
        horizon_out = '/horizon/mask'

        self.thermal_eq_pub = rospy.Publisher(thermal_eq_out, Image, queue_size=1)
        self.horizon_pub = rospy.Publisher(horizon_out, Image, queue_size=1)

        image_sub = message_filters.Subscriber(self.thermal_raw_topic, Image)
        imu_sub = message_filters.Subscriber(self.imu_topic, Imu)

        ts = message_filters.ApproximateTimeSynchronizer([image_sub, imu_sub], 10, 0.02, allow_headerless=True)
        ts.registerCallback(self.thermal_imu_sync_callback)
        rospy.spin()

    def thermal_imu_sync_callback(self, thermal_msg, imu_msg):
        print('Got data {:.2f}'.format(rospy.get_time()))
        rect_img, eq_img = self.thermal_callback(thermal_msg)
        sky_mask = self.horizon_msg_callback(imu_msg, thermal_msg)


        # Publish img for visualization
        # rect_img = self.bridge.cv2_to_imgmsg(rect_img, "16UC1")
        # rect_img.header = thermal_msg.header
        # self.thermal_rect_pub.publish(rect_img)

        # Publish img for visualization
        eq_img = self.bridge.cv2_to_imgmsg(eq_img, "32FC1")
        eq_img.header = thermal_msg.header
        self.thermal_eq_pub.publish(eq_img)

        sky_img = self.bridge.cv2_to_imgmsg(sky_mask, "mono8")
        sky_img.header = thermal_msg.header
        self.horizon_pub.publish(sky_img)

    def thermal_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "mono16")
        
        img = cv2.undistort(img, flir_boson.I, flir_boson.D, None, self.newcameramtx)
        if self.rotate_180:
            img = cv2.rotate(img, cv2.ROTATE_180)

        equalized_img = raw16bit_to_32FC_autoScale(img) 
        equalized_img = exposure.equalize_adapthist(equalized_img, clip_limit=0.02)
        # equalized_img = self.clahe.apply(img) / 2**16
        # print(equalized_img)
        return img, equalized_img.astype(np.float32)
        

    def horizon_msg_callback(self, msg, thermal_msg=None):
        # print(msg.orientation)
        q = msg.orientation
        imu_quat = np.quaternion(q.w, q.x, q.y, q.z)
        q_cam = imu_quat * self.imu2adk_quat

        cam_xyzw = np.array([q_cam.x, q_cam.y, q_cam.z, q_cam.w])
        sky_mask, Xn = estimate_horizon_mask(cam_xyzw, self.new_P, return_points=True, shape=(self.H, self.W))
        sky_mask = cv2.rotate(sky_mask, cv2.ROTATE_180)
        
        # img = self.bridge.imgmsg_to_cv2(thermal_msg, "mono16")
        # img = cv2.undistort(img, flir_boson.I, flir_boson.D, None, self.newcameramtx)
        # img = draw_overlay_and_pts(img, sky_mask, points=Xn, rotate_180=self.rotate_180)

        # sky_img = self.bridge.cv2_to_imgmsg(img, "bgr8")
        # sky_img.header = msg.header
        # self.horizon_pub.publish(sky_img)
        return sky_mask


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rotate-180', action='store_true')  
    args = parser.parse_args()

    print(args)
    try:
        ThermalImuSync(rotate_180=args.rotate_180)
    except rospy.ROSInterruptException:
        pass