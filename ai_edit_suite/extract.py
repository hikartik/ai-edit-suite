# extract.py

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import io, zipfile
from .segmentation import get_class_mask

# 1) Device and model loading at import time
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DeepLabV3 once; if you prefer FCN, you can swap to fcn_resnet101
_DEEPLAB_MODEL = models.segmentation.deeplabv3_resnet101(pretrained=True).to(DEVICE).eval()

# Transform: ToTensor + Normalize
_TO_TENSOR = transforms.ToTensor()
_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Pascal VOC class → index mapping used by torchvision segmentation
OBJ2IDX = {
    "Background": 0, "Aeroplane":1, "bicycle":2, "bird":3, "boat":4, "bottle":5,
    "bus":6, "car":7, "cat":8, "chair":9, "cow":10, "dining table":11,
    "dog":12, "horse":13, "motorbike":14, "person":15, "potted plant":16,
    "sheep":17, "sofa":18, "train":19, "tv/monitor":20
}

def get_class_mask(pil_img: Image.Image, class_idx: int) -> np.ndarray:
    """
    Returns a binary mask (uint8 0 or 255) same size as pil_img where pixels == class_idx.
    Uses DeepLabV3 pretrained. 
    """
    img = pil_img.convert("RGB")
    tensor = _TO_TENSOR(img)
    tensor = _NORMALIZE(tensor).unsqueeze(0).to(DEVICE)  # [1,3,H,W]
    _, _, H, W = tensor.shape

    model = _DEEPLAB_MODEL
    with torch.no_grad():
        out = model(tensor)['out']  # [1,C,H_out,W_out]
        # Upsample to original size
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        cls_map = out.argmax(dim=1)[0].cpu().numpy().astype(np.int32)  # [H,W]
    mask = (cls_map == class_idx).astype(np.uint8) * 255
    return mask

def mask_to_components(mask: np.ndarray) -> list[np.ndarray]:
    """
    Split binary mask (0/255) into connected-component masks (0/255).
    """
    bin_mask = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(bin_mask)
    comps = []
    for lbl in range(1, num_labels):
        comp = (labels == lbl).astype(np.uint8) * 255
        comps.append(comp)
    return comps

def mask_to_bbox(mask: np.ndarray):
    """
    Return (x0, y0, w, h) bounding box of nonzero pixels in mask. If empty, return None.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return (int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1))

def create_rgba_from_mask(pil_img: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Crop pil_img to bounding box of mask and return an RGBA image:
    - The cropped region has RGB from pil_img and alpha from mask.
    """
    bbox = mask_to_bbox(mask)
    if bbox is None:
        raise ValueError("Empty mask: no pixels found for this instance")
    x0, y0, w, h = bbox
    # Crop the RGB image
    crop_rgb = pil_img.crop((x0, y0, x0 + w, y0 + h)).convert("RGBA")
    # Crop the mask
    mask_crop = mask[y0:y0 + h, x0:x0 + w]
    alpha = Image.fromarray(mask_crop).convert("L")
    crop_rgba = Image.new("RGBA", (w, h))
    crop_rgba.paste(crop_rgb, (0, 0))
    crop_rgba.putalpha(alpha)
    return crop_rgba

def extract_single_instance_from_pil(pil_img: Image.Image, class_name: str) -> Image.Image:
    """
    Extract the first instance of class_name from pil_img and return a cropped RGBA PIL image.
    Raises ValueError if class unknown or no instance found.
    """
    name = class_name.strip()
    if name not in OBJ2IDX:
        raise ValueError(f"Unknown class: {name}")
    idx = OBJ2IDX[name]
    # 1) get mask
    mask = get_class_mask(pil_img, idx)
    # 2) split into components
    comps = mask_to_components(mask)
    if not comps:
        raise ValueError(f"No instances of class '{name}' found")
    inst_mask = comps[0]
    # 3) crop RGBA
    rgba = create_rgba_from_mask(pil_img, inst_mask)
    return rgba

def extract_multiple_classes_to_zip_bytes(pil_img: Image.Image, class_names: list[str]) -> bytes:
    """
    For each class in class_names, extract all instances (cropped RGBA) from pil_img,
    pack into a ZIP, and return the ZIP bytes.
    Filenames: "<class>_<i>.png".
    Raises ValueError if any class unknown or no instances found.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        for name in class_names:
            name = name.strip()
            if name not in OBJ2IDX:
                raise ValueError(f"Unknown class: {name}")
            idx = OBJ2IDX[name]
            mask = get_class_mask(pil_img, idx)
            comps = mask_to_components(mask)
            if not comps:
                raise ValueError(f"No instances of class '{name}' found")
            for i, inst_mask in enumerate(comps):
                rgba = create_rgba_from_mask(pil_img, inst_mask)
                img_buf = io.BytesIO()
                rgba.save(img_buf, format="PNG")
                img_buf.seek(0)
                z.writestr(f"{name}_{i}.png", img_buf.read())
    buf.seek(0)
    return buf.getvalue()