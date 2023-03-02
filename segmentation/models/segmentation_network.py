import tqdm
import numpy as np

import torch
import segmentation_models_pytorch as smp

class SegmentationNetwork:

    def __init__(self, weights_path):
        self.model_path = weights_path
        
        self.device = torch.device('cuda:0')
        self.model = smp.FPN(
            encoder_name='timm-mobilenetv3_small_075', 
            encoder_weights=None, 
            in_channels=1, 
            classes=2, 
            activation=None
        )

        self.setup_session_and_warmup()

    def setup_session_and_warmup(self):

        # Input shape is fixed at 512, 640
        weights = torch.load(self.model_path, )
        self.model.load_state_dict(weights, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        fake_input = torch.zeros((1, 1, 512, 640), device=self.device)
        print('Warming up segmentation network...')
        for i in tqdm.tqdm(range(10), total=10):
            self.model(fake_input)
        print('Warmed up!')

    def preprocess_data(self, x):
        x = torch.from_numpy(x).to(self.device).view(1, 1, 512, 640)
        x = 2*x - 1
        return x

    def predict(self, x):
        x = self.preprocess_data(x)
        with torch.inference_mode():
            logits = self.model(x)

        pred_classes = np.argmax(logits.cpu().numpy(), axis=1)
        return np.uint8(255*pred_classes)

    # Update current network with weights of network B.
    def update_weights(self, networkB):
        # Encoder doesn't change. Update decoder only for speed.
        for param_1, param_2 in zip(self.model.decoder.parameters(), networkB.decoder.parameters()):
            param_1.data = param_2.data