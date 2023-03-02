import argparse
import numpy as np

# ROS integration
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import onnxruntime as ort

class OnnxFastSkyCNN:

    def __init__(self, weights_path):
        model_path = weights_path
        
        self.ort_session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider'],
        )

        self.setup_session_and_warmup()
    
    def setup_session_and_warmup(self):

        # Input shape is fixed at 512, 640
        fake_input = np.zeros((1, 1, 512, 640))
        X_ortvalue = ort.OrtValue.ortvalue_from_numpy(fake_input, 'cuda', 0)

        self.io_binding = self.ort_session.io_binding()
        self.io_binding.bind_input(
            name='input', 
            device_type=X_ortvalue.device_name(), 
            device_id=0, 
            element_type=np.float32, 
            shape=X_ortvalue.shape(), 
            buffer_ptr=X_ortvalue.data_ptr()
        )
        self.io_binding.bind_output('output')

        print('Warming up sky segmentation network...')
        self.ort_session.run_with_iobinding(self.io_binding)
        print('Warmed up!')

    def preprocess_data(self, x):
        x = 2*x - 1
        x = np.expand_dims(x, axis=(0, 1))
        return x

    def predict(self, x):
        x = self.preprocess_data(x)
        X_ortvalue = ort.OrtValue.ortvalue_from_numpy(x, 'cuda', 0)
        self.io_binding.bind_ortvalue_input('input', X_ortvalue)
        self.ort_session.run_with_iobinding(self.io_binding)
        
        logits = self.io_binding.copy_outputs_to_cpu()[0]
        pred_classes = np.argmax(logits, axis=1)
        return np.uint8(255*pred_classes)

class SkySegmentationNode:
    
    def __init__(self, args):
        self.args = args
        # Setup FastSCNN segmentation network
        self.onnx_fastcnn = OnnxFastSkyCNN(args.weights_path)
        
        # Subscriber topics
        self.thermal_rect_histeq = '/boson/thermal/image_rect_histeq'

        # CV bridge
        self.bridge = CvBridge()
        self.start_ros()

    def start_ros(self):
        rospy.init_node('sky_segmentation', anonymous=True)
        self.sky_segmentation_pub = rospy.Publisher('/sky/mask', Image, queue_size=1)
        rospy.Subscriber(self.thermal_rect_histeq, Image, self.callback)
        rospy.spin()

    def callback(self, msg):
        print('Got data, doing sky inference {:.2f}'.format(rospy.get_time()))
        img = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        sky_segmentation_mask = self.onnx_fastcnn.predict(img).squeeze()    

        sky_img = self.bridge.cv2_to_imgmsg(sky_segmentation_mask, "mono8")
        sky_img.header = msg.header
        self.sky_segmentation_pub.publish(sky_img)        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-path', default='weights/fast_scnn.onnx')
    args = parser.parse_args()

    try:
        SkySegmentationNode(args)
    except rospy.ROSInterruptException:
        pass