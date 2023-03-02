import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.segmentation_network import SegmentationNetwork
from utils.dataset import ProductionOnlineThermalSegmentationDataset
from utils.utils import postprocess_mask, merge_pseudolabels, momentum_update

# ROS integration
import rospy
import message_filters
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class OnlineTraining:
    
    def __init__(self, args):
        # image width, height
        self.H, self.W = (512, 640)
        self.args = args

        # Setup networks
        self.setup_networks_and_training()

        # Training buffer index
        self.buffer_index = 0
        self.training_description = 'Training time: {:3f}, Label generation time: {:3f}'

        # Subscriber topics
        self.thermal_rect_histeq_topic = '/boson/thermal/image_rect_histeq'
        self.texture_cue_topic = '/cue/texture'
        self.motion_cue_topic = '/cue/motion'
        self.horizon_topic = '/horizon/mask'
        self.sky_topic = '/sky/mask'

        self.sky_horizon_topic = self.horizon_topic
        if self.args.sky_segmentation:
            print('Using sky segmentation...')
            self.sky_horizon_topic = self.sky_topic

        # CV bridge
        self.bridge = CvBridge()

        # Compute constants and start ros
        self.start_ros()

    def setup_networks_and_training(self):
        self.device = torch.device('cuda:0')

        self.inference_network = SegmentationNetwork(weights_path=self.args.weights_path)
        self.momentum_network = SegmentationNetwork(weights_path=self.args.weights_path)
        self.training_network = SegmentationNetwork(weights_path=self.args.weights_path)

        ##########################################################
        # Freeze training network. Momentum and inference networks 
        # not updated so no need to freeze. 
        for param in self.training_network.model.encoder.parameters():
            param.requires_grad = False
        for param in self.training_network.model.decoder.p5.parameters():
            param.requires_grad = False
        for param in self.training_network.model.decoder.p4.parameters():
            param.requires_grad = False
        # for param in self.training_network.model.decoder.p3.parameters():
        #     param.requires_grad = False
        # for param in self.training_network.model.decoder.p2.parameters():
        #     param.requires_grad = False

        #############################################################
        ### Dataset + loading        
        # Setup training buffer + loader
        self.train_buffer = ProductionOnlineThermalSegmentationDataset(
            buffer_size=args.buffer_size, 
            max_crop_width=args.crop_width, 
            epochs=args.epochs
        )
        self.dataloader = DataLoader(
            self.train_buffer, 
            batch_size=self.args.batch_size, 
            num_workers=self.args.num_workers, 
            shuffle=True
        )

        self.optimizer = torch.optim.Adam(self.training_network.model.parameters(), lr=self.args.lr, weight_decay=0.0001)

    def start_ros(self):

        # Create the node
        rospy.init_node('trainer', anonymous=True)

        # Inference publisher/subscribers
        self.segmentation_pub = rospy.Publisher('/segmentation/mask', Image, queue_size=1)
        self.training_pub = rospy.Publisher('/training/times', String, queue_size=1)

        thermal_sub = message_filters.Subscriber(self.thermal_rect_histeq_topic, Image)
        sky_horizon_mask_sub = message_filters.Subscriber(self.sky_horizon_topic, Image)
        texture_cue_sub = message_filters.Subscriber(self.texture_cue_topic, Image)
        motion_cue_sub = message_filters.Subscriber(self.motion_cue_topic, Image)

        if self.args.adapt:
            
            # Use both cues
            if self.args.use_texture and self.args.use_motion:
                ts_train_both = message_filters.ApproximateTimeSynchronizer(
                    [thermal_sub, sky_horizon_mask_sub, texture_cue_sub, motion_cue_sub], 
                    120, 0.1, allow_headerless=False)
                
                ts_train_both.registerCallback(self.texture_motion_callback)

            elif self.args.use_texture:
                ts_train_texture = message_filters.ApproximateTimeSynchronizer(
                    [thermal_sub, sky_horizon_mask_sub, texture_cue_sub], 
                    30, 0.01, allow_headerless=False)
                
                ts_train_texture.registerCallback(self.texture_callback)

            elif self.args.use_motion:
                ts_train_motion = message_filters.ApproximateTimeSynchronizer(
                    [thermal_sub, sky_horizon_mask_sub, motion_cue_sub], 
                    120, 0.1, allow_headerless=False)
                
                ts_train_motion.registerCallback(self.motion_callback)

            
        ts_inf = message_filters.ApproximateTimeSynchronizer([thermal_sub, sky_horizon_mask_sub], 10, 0.01, allow_headerless=False)
        ts_inf.registerCallback(self.inference_callback)
        rospy.spin()


    # Intermediary callbacks
    def texture_callback(self, thermal_msg, horizon_msg, texture_msg):
        self.training_callback(thermal_msg, horizon_msg, texture_msg=texture_msg)

    def motion_callback(self, thermal_msg, horizon_msg, motion_msg):
        self.training_callback(thermal_msg, horizon_msg, motion_msg=motion_msg)

    def texture_motion_callback(self, thermal_msg, horizon_msg, texture_msg, motion_msg):
        self.training_callback(thermal_msg, horizon_msg, texture_msg=texture_msg, motion_msg=motion_msg)


    def inference_callback(self, thermal_msg, horizon_msg):
        # print('Got data, doing water inference {:.2f}'.format(rospy.get_time()))
        img = self.bridge.imgmsg_to_cv2(thermal_msg, "32FC1")
        water_segmentation = self.inference_network.predict(img).squeeze()    
        
        sky_mask = self.bridge.imgmsg_to_cv2(horizon_msg, "mono8")
        water_segmentation[sky_mask == 255] = 0

        if self.args.postprocess:
            water_segmentation = postprocess_mask(water_segmentation)

        water_img = self.bridge.cv2_to_imgmsg(water_segmentation, "mono8")
        water_img.header = thermal_msg.header
        self.segmentation_pub.publish(water_img)   

    def training_callback(self, thermal_msg, horizon_msg, texture_msg=None, motion_msg=None):
        print('Got training data {:.2f}'.format(rospy.get_time()))

        img = self.bridge.imgmsg_to_cv2(thermal_msg, "32FC1")
        sky_horizon_cue = self.bridge.imgmsg_to_cv2(horizon_msg, "mono8")

        texture_cue, motion_cue = None, None
        if texture_msg is not None:
            texture_cue = self.bridge.imgmsg_to_cv2(texture_msg, "32FC1")
        if motion_msg is not None:
            motion_cue = self.bridge.imgmsg_to_cv2(motion_msg, "32FC1")

        self.train_buffer.update_samples(img, sky_horizon_cue, self.buffer_index, texture_cue=texture_cue, motion_cue=motion_cue)
        self.buffer_index += 1

        # LIFO order: replace earliest seen sample
        if self.buffer_index == self.args.buffer_size: 
            self.buffer_index = 0
            
            self.online_train()
            
            start = time.time()
            self.inference_network.update_weights(self.training_network.model)
            # self.buffer_index = 4
            # self.train_buffer.drop_and_coalesce()
            end = time.time()
            
            print("Weight transfer time: ", end - start)
            
    def online_train(self):
        print('Training...')
        train_start = time.time()
        momentum_update_rate = self.args.momentum_update_rate
        
        # ###########################################################################
        # ### One iteration of online-SSL 
        # ###########################################################################
        for i, train_batch in enumerate(self.dataloader):
            x = train_batch[0].to(self.device)              # Always here
            sky_horizon_cue = train_batch[1].to(self.device)    # Always here

            if self.args.use_texture: 
                texture_cue = train_batch[2].to(self.device)    # Sometimes
            if self.args.use_motion:
                motion_cue = train_batch[3].to(self.device)     # Sometimes
            
            label_generation_start = time.time()
            final_label = None
            with torch.no_grad():
                mask = self.momentum_network.model(x)

                valid_mask = torch.ones_like(mask)

                if self.args.use_texture and self.args.use_motion:
                    final_label, valid_mask = merge_pseudolabels(
                        mask, 
                        sift_pseudolabel=texture_cue, 
                        flow_pseudolabel=motion_cue, 
                        sky_pseudolabel=sky_horizon_cue)
                    
                elif self.args.use_texture:
                    final_label, valid_mask = merge_pseudolabels(
                        mask, 
                        sift_pseudolabel=texture_cue, 
                        sky_pseudolabel=sky_horizon_cue)
                
                elif self.args.use_motion:
                    final_label, valid_mask = merge_pseudolabels(
                        mask, 
                        flow_pseudolabel=motion_cue, 
                        sky_pseudolabel=sky_horizon_cue)
                

            label_generation_end = time.time()

            logits = self.training_network.model(x)
            prob = torch.sigmoid(logits)*valid_mask
            train_loss = F.binary_cross_entropy(prob, final_label*valid_mask)

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            # Goes outside loop?
            if i % int(self.args.buffer_size / self.args.batch_size) == 0:
                print('Momentum update...')
                momentum_update(
                    self.momentum_network.model, 
                    self.training_network.model, 
                    momentum_rate=momentum_update_rate
                )
        print('Done training...')
        train_end = time.time()

        training_time = train_end - train_start
        label_time = label_generation_end - label_generation_start
        description = self.training_description.format(training_time, label_time)
        self.training_pub.publish(description)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--buffer-size', default=8, type=int)
    parser.add_argument('--crop-width', default=512, type=int)
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num-workers', default=4, type=int)

    parser.add_argument('--weights-path', default='weights/mobilenetv3_fpn.ckpt')

    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--momentum-update-rate', default=0.3, type=float)

    parser.add_argument('--use-texture', action='store_true')
    parser.add_argument('--use-motion', action='store_true')

    parser.add_argument('--sky-segmentation', action='store_true')
    
    parser.add_argument('--postprocess', action='store_true')
    parser.add_argument('--adapt', action='store_true')

    args = parser.parse_args()


    try:
        OnlineTraining(args)
    except rospy.ROSInterruptException:
        pass