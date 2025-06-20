# predict.py

import tempfile
import os
from PIL import Image
import torch
import random
import cv2
import numpy as np

from .src.config import Config
from .src.edge_connect import EdgeConnect
# Import Dataset class to rebuild test_dataset
from .src.dataset import Dataset

# Maps object to index
obj2idx = {
    "Background": 0, "Aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5,
    "bus": 6, "car": 7, "cat": 8, "chair": 9, "cow": 10, "dining table": 11,
    "dog": 12, "horse": 13, "motorbike": 14, "person": 15, "potted plant": 16,
    "sheep": 17, "sofa": 18, "train": 19, "tv/monitor": 20
}

# Device once
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare Config once
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory containing predict.py
# Assuming config.yml is at <project_root>/checkpoints/config.yml and predict.py is in <project_root>/...
_CONFIG_PATH = os.path.join(_BASE_DIR, "checkpoints", "config.yml")
if not os.path.exists(_CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found at {_CONFIG_PATH}")

_config = Config(_CONFIG_PATH)
# Set common config fields once
_config.MODE = 2
_config.MODEL = 3
_config.SEG_DEVICE = _DEVICE
_config.INPUT_SIZE = 256
# RESULTS will be set per-call or can remain None
_config.RESULTS = None

# Global model placeholder
_model = None

def init_model():
    global _model
    if _model is None:
        # Build and load EdgeConnect once
        _model = EdgeConnect(_config)
        _model.load()
    return _model

# Alias for init_model so init_inpainter can be imported
def init_inpainter():
    """
    Alias to initialize and return the EdgeConnect model.
    """
    return init_model()

def remove_objects_from_pil(pil_img: Image.Image, class_names: list[str]) -> Image.Image:
    """
    Remove given classes from the PIL image using EdgeConnect inpainting.
    Returns a new PIL RGB image.
    """
    # 1. Validate class_names and map to indices
    indices = []
    for name in class_names:
        name = name.strip()
        if name not in obj2idx:
            raise ValueError(f"Unknown class: {name}")
        indices.append(obj2idx[name])
    if not indices:
        raise ValueError("No valid class names provided")

    # 2. Save pil_img to a temporary file
    tmpdir = tempfile.mkdtemp()
    in_path = os.path.join(tmpdir, "input.png")
    pil_img.save(in_path)

    # 3. Prepare config for this inference
    _config.TEST_FLIST = in_path
    _config.TEST_EDGE_FLIST = in_path
    _config.OBJECTS = indices
    _config.RESULTS = tmpdir

    # 4. Seeds & threads
    cv2.setNumThreads(0)
    torch.manual_seed(_config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_config.SEED)
    np.random.seed(_config.SEED)
    random.seed(_config.SEED)

    # 5. Initialize (or get) the model
    model = init_model()

    # 6. Rebuild model.test_dataset so it uses updated TEST_FLIST/TEST_EDGE_FLIST
    try:
        # Dataset(config, flist, edge_flist, augment=False, training=False)
        model.test_dataset = Dataset(_config, _config.TEST_FLIST, _config.TEST_EDGE_FLIST, augment=False, training=False)
    except Exception as e:
        raise RuntimeError(f"Failed to build test_dataset: {e}")

    # 7. Run inference
    try:
        output = model.test()
    except Exception as e:
        # propagate so endpoint can return a clear error
        raise RuntimeError(f"Inference failed: {e}")

    if output is None:
        raise RuntimeError("Inference returned no output")

    # 8. Convert output to PIL Image
    if isinstance(output, torch.Tensor):
        arr = output.cpu().numpy()
        # Handle shapes: [1, C, H, W], [C, H, W], or [H, W, C]
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3:
            # If shape [C, H, W], transpose to [H, W, C]
            if arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(arr).convert("RGB")
    else:
        out_img = output  # assume PIL.Image

    return out_img
