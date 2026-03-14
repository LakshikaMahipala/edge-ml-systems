import numpy as np
import cv2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_bgr_to_nchw_float(frame_bgr, input_size: int) -> np.ndarray:
    """
    frame_bgr: uint8 HxWx3 (OpenCV BGR)
    return: float32 1x3xHxW normalized (ImageNet)
    """
    frame = cv2.resize(frame_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = frame_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = np.expand_dims(x, axis=0)   # NCHW
    return x.astype(np.float32)
