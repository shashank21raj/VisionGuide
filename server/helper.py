import numpy as np
import cv2
from config import DEPTH_WEIGHT,FLOW_WEIGHT

def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = np.float32(arr)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.float16)
    normalized = (arr - mn) / (mx - mn)
    return np.clip(normalized, 0.0, 1.0).astype(np.float16)

def compute_optical_flow(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 2, 15, 1, 5, 1.1, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return normalize_map(mag)

def combine_maps(d: np.ndarray, f: np.ndarray) -> np.ndarray:
    return normalize_map(DEPTH_WEIGHT * d + FLOW_WEIGHT * f)

def box_mean_heat(heat_map: np.ndarray, box: tuple) -> float:
    x1, y1, x2, y2 = map(int, box)
    h, w = heat_map.shape
    x1, x2 = np.clip([x1, x2], 0, w-1)
    y1, y2 = np.clip([y1, y2], 0, h-1)
    return 0.0 if x2 <= x1 or y2 <= y1 else float(heat_map[y1:y2, x1:x2].mean())