import numpy as np

def raw16bit_to_32FC_autoScale(img):
    img = img - np.percentile(img, 1)
    img = img / np.percentile(img, 99)
    img = np.clip(img, 0, 1)
    
    return img.astype(np.float32)