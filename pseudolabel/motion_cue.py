import numpy as np
import cv2
import matplotlib.pyplot as plt

# ROS integration
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class FlowPseudoLabeler:
    
    def __init__(self, plot=False):
        self.plot = False

        maxFeatures = 500
        self.orb = cv2.ORB_create(maxFeatures)
        self.matcher = cv2.BFMatcher()

    def label(self, img_curr, img_prev, horizon_mask=None):
        img_curr = np.uint8(255*img_curr)
        img_prev = np.uint8(255*img_prev)

        (kps_curr, descs_curr) = self.orb.detectAndCompute(img_curr, None)
        (kps_prev, descs_prev) = self.orb.detectAndCompute(img_prev, None)

        (h, w) = img_curr.shape[:2]
        segm = -np.ones((h, w))
        try:
            matches = self.matcher.knnMatch(descs_prev, descs_curr, k=2)
        
            good = []
            for m,n in matches:
                curr = np.asarray(kps_curr[m.trainIdx].pt)
                prev = np.asarray(kps_prev[m.queryIdx].pt) 

                if m.distance < 0.75*n.distance and np.linalg.norm(curr - prev) < 10:
                    good.append([m])

            pts_curr = np.zeros((len(matches), 2), dtype="float")
            pts_prev = np.zeros((len(matches), 2), dtype="float")
            for i, m in enumerate(good):
                pts_curr[i] = kps_curr[m[0].trainIdx].pt
                pts_prev[i] = kps_prev[m[0].queryIdx].pt

            (H_matrix, mask) = cv2.findHomography(pts_prev, pts_curr, method=cv2.RANSAC)

            # use the homography matrix to align the images
            aligned_prev = cv2.warpPerspective(img_prev, H_matrix, (w, h))

            box = np.array(
                [[0, 0, 1],
                [0, w, 1], 
                [h, 0, 1],
                [h, w, 1]]).T

            crop_box = H_matrix @ box
            crop_box /= crop_box[2,:]

            crop_box = np.int32(crop_box)
            x1 = max(crop_box[0, 0], 0)
            x2 = min(crop_box[1, 3], w)
            y1 = max(crop_box[1, 0], 0)
            y2 = min(crop_box[0, 3], h)

            aligned_prev_crop = aligned_prev[y1:y2, x1:x2]
            img_curr_crop = img_curr[y1:y2, x1:x2]

            flow = cv2.calcOpticalFlowFarneback(aligned_prev_crop, img_curr_crop, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            p75 = np.percentile(flow_magnitude, 75)
            cropped_img_segm = np.zeros_like(flow_magnitude)
            if p75 > 1e-1:
                cropped_img_segm = np.clip(flow_magnitude / p75, 0, 1)

            segm = np.zeros((h, w)) + 0.5
            segm[y1:y2, x1:x2] = cropped_img_segm
            segm = cv2.GaussianBlur(segm, ksize=(45, 45), sigmaX=10, sigmaY=10, borderType=cv2.BORDER_REPLICATE)
            
            if self.plot:
                img1 = np.hstack([aligned_prev_crop, img_curr_crop])
                img1 = cv2.resize(img1, (0, 0), fx=2, fy=2) 
                cv2.imwrite('temp_visuals/warped_crops.png', img1)

                # --- Display
                fig, (ax0, ax1) = plt.subplots(1, 2)
                ax0.imshow(img_curr_crop, cmap='gray')
                ax0.set_axis_off()
                ax1.imshow(flow_magnitude, cmap='gray')
                ax1.set_axis_off()
                fig.tight_layout()
                plt.savefig('temp_visuals/flow.png')

        except Exception as e:
            print(e)
            print('homography not found: no flow used.')
            pass
        
        return segm.astype(np.float32)

class MotionCueNode:
    
    def __init__(self):
        
        self.pseudolabeler = FlowPseudoLabeler()
        # Subscriber topics
        self.thermal_rect_histeq = '/boson/thermal/image_rect_histeq'
        self.horizon_topic = '/horizon/mask'

        self.prev_img = None
        self.prev_pctl = None

        # CV bridge
        self.bridge = CvBridge()
        self.start_ros()

    def start_ros(self):
        print('Starting motion cue node')
        rospy.init_node('motion_cue', anonymous=True)
        self.motion_label_pub = rospy.Publisher('/cue/motion', Image, queue_size=1)

        thermal_sub = message_filters.Subscriber(self.thermal_rect_histeq, Image)
        horizon_mask_sub = message_filters.Subscriber(self.horizon_topic, Image)
        
        motion_subscriber = message_filters.ApproximateTimeSynchronizer(
            [thermal_sub, horizon_mask_sub], 
            10, 0.01, allow_headerless=False
        )
        
        motion_subscriber.registerCallback(self.callback)

        rospy.spin()

    def callback(self, thermal_msg, horizon_msg):
        print('Got data, generating motion cue {:.2f}'.format(rospy.get_time()))
        img = self.bridge.imgmsg_to_cv2(thermal_msg, "32FC1")
        img = np.uint8(255*img)
        curr_img_orig = cv2.resize(img, (320, 256))
        curr_pctl = np.percentile(curr_img_orig, [1, 99])

        horizon_mask = self.bridge.imgmsg_to_cv2(horizon_msg, "mono8")
        horizon_mask = cv2.resize(horizon_mask, (320, 256))

        prev_img = self.prev_img
        if prev_img is not None:

            p1 = max(curr_pctl[0], self.prev_pctl[0])
            p99 = min(curr_pctl[1], self.prev_pctl[1])

            curr_img = np.clip((curr_img_orig - p1) / (p99 - p1), 0, 1) 
            prev_img = np.clip((prev_img - p1) / (p99 - p1), 0, 1) 

            mask = self.pseudolabeler.label(curr_img, prev_img, horizon_mask)
            mask = cv2.resize(mask, (640, 512))

            land_img = self.bridge.cv2_to_imgmsg(mask, "32FC1")
            land_img.header = thermal_msg.header
            self.motion_label_pub.publish(land_img)     

        self.prev_img = curr_img_orig   
        self.prev_pctl = curr_pctl

if __name__ == '__main__':
    try:
        MotionCueNode()
    except rospy.ROSInterruptException:
        pass