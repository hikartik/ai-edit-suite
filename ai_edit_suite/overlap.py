# overlap.py

from PIL import Image
import numpy as np

# Import helpers from extract.py
from .extract import (
    get_class_mask,
    mask_to_components,
    create_rgba_from_mask,
    OBJ2IDX,
    mask_to_bbox,
)
def overlap_single_class_from_pil(pil_img1: Image.Image, pil_img2: Image.Image, class_name: str) -> Image.Image:
    """
    Extract all instances of class_name from pil_img1 and overlay them onto pil_img2.
    Returns a new PIL RGB image.
    If pil_img2 size differs from pil_img1, pil_img2 is resized to pil_img1.size.
    """
    name = class_name.strip()
    if not name:
        raise ValueError("class_name is empty")
    if name not in OBJ2IDX:
        raise ValueError(f"Unknown class: {name}")
    # Ensure same size: resize image2 to image1 size if needed
    size1 = pil_img1.size  # (width, height)
    if pil_img2.size != size1:
        pil_base = pil_img2.resize(size1, Image.LANCZOS)
    else:
        pil_base = pil_img2
    # Prepare base RGBA
    base_rgba = pil_base.convert("RGBA")
    # Create a transparent layer
    layer = Image.new("RGBA", size1)
    # Get mask and components from image1
    idx = OBJ2IDX[name]
    mask = get_class_mask(pil_img1, idx)  # np.ndarray HxW 0/255
    comps = mask_to_components(mask)
    if not comps:
        # No instances: return base unchanged (or raise error)
        # Here we choose to return base unchanged
        return base_rgba.convert("RGB")
    # For each instance, crop and paste
    for inst_mask in comps:
        try:
            patch_rgba = create_rgba_from_mask(pil_img1, inst_mask)
        except ValueError:
            continue
        # Determine bbox to get paste position
        # mask_to_components returns masks same size as image; create_rgba crops to bbox
        # Need top-left coords:
        from .extract import mask_to_bbox
        bbox = mask_to_bbox(inst_mask)
        if bbox is None:
            continue
        x0, y0, w, h = bbox
        # Paste patch onto layer at (x0, y0)
        layer.paste(patch_rgba, (x0, y0), patch_rgba)
    # Composite layer over base
    out = Image.alpha_composite(base_rgba, layer).convert("RGB")
    return out

def overlap_classes_from_pil(pil_img1: Image.Image, pil_img2: Image.Image, class_names: list[str]) -> Image.Image:
    """
    For each class in class_names, extract all instances from pil_img1 and overlay them onto pil_img2.
    Applies sequentially: the overlays accumulate on the base image.
    If pil_img2 size differs, it is resized once to match pil_img1.
    """
    # Ensure same size
    size1 = pil_img1.size
    if pil_img2.size != size1:
        base = pil_img2.resize(size1, Image.LANCZOS)
    else:
        base = pil_img2
    # Start with RGBA base
    current = base.convert("RGBA")
    for name in class_names:
        # Overlay class on current
        # We extract from pil_img1 but paste onto current
        overlaid = overlap_single_class_from_pil(pil_img1, current.convert("RGB"), name)
        # Prepare for next: convert back to RGBA
        current = overlaid.convert("RGBA")
    return current.convert("RGB")
