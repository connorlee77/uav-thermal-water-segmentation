import torch
import numpy as np
import cv2
import albumentations as A

class ProductionOnlineThermalSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, buffer_size=8, max_crop_width=512, epochs=4):
        super().__init__()

        self.data = np.zeros((buffer_size, 4, 512, 640))
        self.epochs = epochs

        self.data_samples = buffer_size
        self.iterations = self.data_samples * self.epochs

        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            # A.LongestMaxSize(max_size=max_crop_width, always_apply=True),
            A.RandomCrop(320, 320, always_apply=True)
        ], additional_targets={
            'mask1' : 'mask', 
            'mask2' : 'mask'
            }
        )
        
    def update_samples(self, img, horizon_cue, idx, texture_cue=None, motion_cue=None):
        self.data[idx, 0] = img
        self.data[idx, 1] = horizon_cue

        if texture_cue is not None:
            self.data[idx, 2] = texture_cue   

        if motion_cue is not None:
            self.data[idx, 3] = motion_cue   
    
    def drop_and_coalesce(self):
        self.data[:4] = self.data[4:]


    def __len__(self):
        return self.iterations

    def __getitem__(self, index):
        sample = self.data[index % self.data_samples]
        img, horizon, texture_segm, motion_segm = sample[0], sample[1], sample[2], sample[3]

        augmented_data = self.train_transform(image=img, mask=horizon, mask1=texture_segm, mask2=motion_segm)
        
        horizon_tensor = torch.from_numpy(augmented_data['mask']).bool()
        texture_segm_tensor = torch.from_numpy(augmented_data['mask1']).float()
        motion_segm_tensor = torch.from_numpy(augmented_data['mask2']).float()
        
        aug_img = augmented_data['image'] 
        img_tensor = 2*torch.from_numpy(aug_img).unsqueeze(0).float() - 1

        return img_tensor, horizon_tensor, texture_segm_tensor, motion_segm_tensor
