# segmentation.py

import torch
import torch.nn.functional as F
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals to hold models
_seg_models = {}

# Transform
_TO_TENSOR = transforms.ToTensor()
_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def load_segmentation_model(seg_net: int = 0):
    """
    seg_net=0 → DeepLabV3
    seg_net=1 → FCN
    Loads model once into _seg_models and returns it.
    """
    key = f"seg_{seg_net}"
    if key not in _seg_models:
        if seg_net == 1:
            model = models.segmentation.fcn_resnet101(pretrained=True)
        else:
            model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        model = model.to(DEVICE).eval()
        _seg_models[key] = model
    return _seg_models[key]

def get_class_mask(pil_img, class_idx: int, seg_net: int = 0):
    """
    Returns binary mask (HxW uint8 0/255) where pixels == class_idx.
    """
    img = pil_img.convert("RGB")
    tensor = _TO_TENSOR(img)
    tensor = _NORMALIZE(tensor).unsqueeze(0).to(DEVICE)  # [1,3,H,W]
    _, _, H, W = tensor.shape
    model = load_segmentation_model(seg_net)
    with torch.no_grad():
        out = model(tensor)['out']
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        cls_map = out.argmax(dim=1)[0].cpu().numpy().astype(int)
    mask = (cls_map == class_idx).astype('uint8') * 255
    return mask
