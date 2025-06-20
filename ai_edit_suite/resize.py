# resize.py

from PIL import Image
import numpy as np
# Import segmentation helpers from extract.py
from .extract import (
    get_class_mask,
    mask_to_components,
    create_rgba_from_mask,
    OBJ2IDX,
    mask_to_bbox,
)

# Import your removal/inpainting function
from .object_eraser.predict import remove_objects_from_pil

def resize_single_instance_from_pil(pil_img: Image.Image, class_name: str, scale: float) -> Image.Image:
    """
    Resize the first instance of `class_name` in `pil_img` by `scale`, after removing it and inpainting the background.
    Steps:
      1) Validate class and scale.
      2) Get mask for the class on original image.
      3) Split into connected components; pick the first.
      4) Remove/inpaint via remove_objects_from_pil.
      5) Ensure bg_img matches original size.
      6) Crop original image to instance bbox → RGBA patch.
      7) Resize patch by scale.
      8) Composite resized patch onto the inpainted background at the original center.
    Returns: PIL RGB image.
    Raises ValueError if class unknown, no instance, or scale invalid.
    """
    name = class_name.strip()
    if not name:
        raise ValueError("class_name is empty")
    if name not in OBJ2IDX:
        raise ValueError(f"Unknown class: {name}")
    if scale <= 0:
        raise ValueError(f"Scale must be > 0, got {scale}")

    # 1) segmentation mask on original
    idx = OBJ2IDX[name]
    mask = get_class_mask(pil_img, idx)  # np.ndarray uint8 0/255
    comps = mask_to_components(mask)
    if not comps:
        raise ValueError(f"No instances of class '{name}' found")
    inst_mask = comps[0]

    # 2) remove/inpaint background
    try:
        bg_img = remove_objects_from_pil(pil_img, [name])
    except Exception as e:
        raise RuntimeError(f"Error during removal/inpainting: {e}")
    # bg_img may not match pil_img size; ensure it does:
    if bg_img.size != pil_img.size:
        bg_img = bg_img.resize(pil_img.size, Image.LANCZOS)

    # 3) crop original to RGBA patch
    try:
        patch_rgba = create_rgba_from_mask(pil_img, inst_mask)
    except Exception as e:
        raise RuntimeError(f"Error cropping instance: {e}")

    # 4) determine original center of bbox
    bbox = mask_to_bbox(inst_mask)
    if bbox is None:
        raise ValueError("Empty instance mask after cropping")
    x0, y0, w0, h0 = bbox
    cx = x0 + w0 // 2
    cy = y0 + h0 // 2

    # 5) resize patch
    w, h = patch_rgba.size
    new_w = int(w * scale)
    new_h = int(h * scale)
    if new_w < 1 or new_h < 1:
        raise ValueError(f"Scale {scale} too small: resulting patch size {new_w}x{new_h}")
    patch_small = patch_rgba.resize((new_w, new_h), Image.LANCZOS)

    # 6) composite onto bg_img at center
    paste_x = cx - new_w // 2
    paste_y = cy - new_h // 2

    # Composite preserving alpha
    bg = bg_img.convert("RGBA")
    layer = Image.new("RGBA", bg.size)
    layer.paste(patch_small, (paste_x, paste_y), patch_small)
    out = Image.alpha_composite(bg, layer).convert("RGB")
    return out

def resize_multiple_classes_from_pil(pil_img: Image.Image, class_names: list[str], scales: list[float]) -> Image.Image:
    """
    Sequentially resize first instance of each class in class_names by corresponding scale in scales.
    Returns final PIL RGB image.
    """
    if len(class_names) != len(scales):
        raise ValueError("class_names and scales must have the same length")
    current = pil_img
    for name, scale in zip(class_names, scales):
        current = resize_single_instance_from_pil(current, name, scale)
    return current
