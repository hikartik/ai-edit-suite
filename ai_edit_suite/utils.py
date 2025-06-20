# utils.py

import numpy as np
import cv2
from PIL import Image

def mask_to_components(mask: np.ndarray) -> list[np.ndarray]:
    """
    Split binary mask (0/255) into connected-component masks (0/255).
    Returns a list of masks (uint8 arrays) same shape as input.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    bin_mask = (mask > 0).astype(np.uint8)
    # OpenCV connectedComponents: background=0, labels 1...N
    num_labels, labels = cv2.connectedComponents(bin_mask)
    comps = []
    for lbl in range(1, num_labels):
        comp = (labels == lbl).astype(np.uint8) * 255
        comps.append(comp)
    return comps

def mask_to_bbox(mask: np.ndarray):
    """
    Given binary mask (0/255), return bounding box (x0, y0, width, height).
    If mask empty, return None.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return (int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1))

def create_rgba_from_mask(pil_img: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Crop pil_img to bounding box of mask and produce an RGBA image with alpha = mask.
    - pil_img: PIL Image in RGB, size matches mask.shape (height, width).
    - mask: np.ndarray uint8 0 or 255, same size as pil_img.
    Returns: PIL Image in RGBA mode, size = bounding-box dimensions.
    """
    bbox = mask_to_bbox(mask)
    if bbox is None:
        raise ValueError("Empty mask: no pixels to extract")
    x0, y0, w, h = bbox
    # Crop the RGB image
    crop_rgb = pil_img.crop((x0, y0, x0 + w, y0 + h)).convert("RGBA")
    # Crop the mask
    mask_crop = mask[y0:y0 + h, x0:x0 + w]
    alpha = Image.fromarray(mask_crop).convert("L")
    # Create RGBA image
    crop_rgba = Image.new("RGBA", (w, h))
    crop_rgba.paste(crop_rgb, (0, 0))
    crop_rgba.putalpha(alpha)
    return crop_rgba

def composite_rgba_on_background(bg_img: Image.Image, fg_rgba: Image.Image, position: tuple[int,int]) -> Image.Image:
    """
    Paste fg_rgba (with alpha channel) onto bg_img at given position (x,y).
    - bg_img: PIL Image in RGB or RGBA.
    - fg_rgba: PIL Image in RGBA.
    Returns: composited PIL Image in RGB.
    """
    # Ensure bg is RGBA
    bg = bg_img.convert("RGBA")
    layer = Image.new("RGBA", bg.size)
    x, y = position
    layer.paste(fg_rgba, (int(x), int(y)), fg_rgba)
    out = Image.alpha_composite(bg, layer).convert("RGB")
    return out
