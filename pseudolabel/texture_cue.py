import cv2
from collections import Counter
from skimage.segmentation import slic, find_boundaries
from skimage.morphology import disk, binary_dilation
from skimage.util import img_as_float
import numpy as np

# ROS integration
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class SiftPseudoLabeler:
    
    def __init__(self, max_pts=10, density=True, plot=False):
        self.sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10, nOctaveLayers=3, sigma=1.6)
        self.fast = cv2.FastFeatureDetector_create(threshold=20, type=2)
        self.max_pts = max_pts # threshhold
        self.density = density
        self.plot = plot

    def label(self, img):

        kp2 = self.sift.detect(img, None)

        curr_segs = slic(img_as_float(img), n_segments=100, slic_zero=False, compactness=0.1, sigma=1, start_label=1)
        slic_boundaries = find_boundaries(curr_segs, mode='thick', connectivity=1)
        slic_boundaries = binary_dilation(slic_boundaries, disk(1))

        keyPtCount = Counter()
        seen_kps = set()
        for kp in kp2:
            pt_tuple = kp.pt
            if pt_tuple not in seen_kps:
                seen_kps.add(pt_tuple)
                
                pt_ry = int(np.rint(pt_tuple[1]))
                pt_cx = int(np.rint(pt_tuple[0]))
                keyPtOnBoundary = slic_boundaries[pt_ry, pt_cx] == 1
                if not keyPtOnBoundary:
                    keyPtCount[curr_segs[pt_ry, pt_cx]] += 1

        img_segm = np.zeros(curr_segs.shape, dtype=np.float32) 
        max_seg_idx = np.max(curr_segs)
        idx = 1
        while idx < max_seg_idx + 1:
            if self.density:
                img_segm[curr_segs == idx] = 1 - np.clip(keyPtCount[idx] / self.max_pts, 0, 1)
            else:
                img_segm[curr_segs == idx] = (keyPtCount[idx] < self.max_pts)
            idx += 1
  
        img_segm = cv2.GaussianBlur(img_segm, (45, 45), 11)
        superpixel_img = None

        return img_segm, superpixel_img

class TextureCueNode:
    
    def __init__(self):
        
        self.pseudolabeler = SiftPseudoLabeler()
        # Subscriber topics
        self.thermal_rect_histeq = '/boson/thermal/image_rect_histeq'

        # CV bridge
        self.bridge = CvBridge()
        self.start_ros()

    def start_ros(self):
        rospy.init_node('texture_cue', anonymous=True)
        self.texture_label_pub = rospy.Publisher('/cue/texture', Image, queue_size=1)
        rospy.Subscriber(self.thermal_rect_histeq, Image, self.callback)
        rospy.spin()

    def callback(self, msg):
        print('Got data, generating texture cue {:.2f}'.format(rospy.get_time()))
        img = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        img = np.uint8(255*img)
        img_small = cv2.resize(img, (320, 256))

        land_mask, debug_img = self.pseudolabeler.label(img_small)
        land_mask = cv2.resize(land_mask, (640, 512))

        land_img = self.bridge.cv2_to_imgmsg(land_mask, "32FC1")
        land_img.header = msg.header
        self.texture_label_pub.publish(land_img)        


if __name__ == '__main__':
    try:
        TextureCueNode()
    except rospy.ROSInterruptException:
        pass