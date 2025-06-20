from torchvision import models
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import numpy as np
from imageio import imread
from skimage.color import rgb2gray, gray2rgb
import cv2

# Load segmentation models
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()
# Ensure pretrained argument consistency for DeepLab
dlab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()


# Determine resample filter
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

def decode_segmap(image, objects, nc=21):
    """
    Given a 2D array `image` of class indices, and a list `objects` of class indices to keep,
    produce a mask where selected classes are 255 and others 0.
    """
    r = np.zeros_like(image, dtype=np.uint8)
    for l in objects:
        r[image == l] = 255
    return r

def fill_gaps(values):
    searchval = [255, 0, 255]
    searchval2 = [255, 0, 0, 255]
    idx = np.where((values[:-2] == searchval[0]) & (values[1:-1] == searchval[1]) & (values[2:] == searchval[2]))[0] + 1
    idx2 = np.where((values[:-3] == searchval2[0]) & (values[1:-2] == searchval2[1]) & (values[2:-1] == searchval2[2]) & (values[3:] == searchval2[3]))[0] + 1
    idx3 = idx2 + 1
    if idx.size or idx2.size:
        new_idxs = np.concatenate([idx, idx2, idx3])
    else:
        new_idxs = np.array([], dtype=int)
    values[new_idxs] = 255
    return values

def fill_gaps2(values):
    searchval = [0, 255]
    idx = np.where((values[:-1] == searchval[0]) & (values[1:] == searchval[1]))[0]
    idx2 = idx + 1
    if idx.size:
        new_idxs = np.concatenate([idx, idx2])
    else:
        new_idxs = np.array([], dtype=int)
    values[new_idxs] = 255
    return values

def remove_patch_og(real_img, mask):
    """
    real_img: numpy array shape (H, W, C) or (H, W)
    mask: numpy array shape (H, W), values 0 or 255
    Sets pixels where mask==255 to white (255) across channels.
    """
    og_data = real_img.copy()
    idx = (mask == 255)
    if og_data.ndim == 3:
        og_data[idx, :] = 255
    else:
        og_data[idx] = 255
    return og_data

def segmentor(seg_net, img, dev, objects):
    """
    Run segmentation on PIL Image `img`, then create mask for `objects`,
    resize original image to mask size, and remove masked regions.
    Returns numpy arrays (og_img, mask).
    """
    # Choose model and move to device
    net = fcn if seg_net == 1 else dlab
    net = net.to(dev).eval()

    # Prepare transform
    resize_size = 400 if dev == 'cuda' else 680
    trf = T.Compose([
        T.Resize(resize_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Ensure img is PIL Image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    inp = trf(img).unsqueeze(0).to(dev)

    with torch.no_grad():
        out = net(inp)['out']
    om = torch.argmax(out.squeeze(0), dim=0).cpu().numpy()
    mask = decode_segmap(om, objects)

    # Resize original image to mask size
    height, width = mask.shape
    img_resized = img.resize((width, height), RESAMPLE)
    img_arr = np.array(img_resized)

    # Remove masked patch
    og_img = remove_patch_og(img_arr, mask)
    return og_img, mask
