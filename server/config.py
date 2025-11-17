import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For resolution
INFER_W, INFER_H = 320, 240
DISPLAY_W, DISPLAY_H = 480, 360

# for Inference frequency
DEPTH_EVERY = 5
DETECT_EVERY = 1  # Run YOLO every frame for debugging

# for Heat map weights
DEPTH_WEIGHT = 0.80
FLOW_WEIGHT = 0.20
SMOOTHING_ALPHA = 0.70

# for Safe path thresholds
SAFE_HEAT_MAX = 0.65
MIN_FREE_WIDTH_PCT = 0.38
MIN_FREE_HEIGHT = 135
MIN_SAFE_THRESHOLD = 0.90

# for Detection thresholds
FOCUS_CLASSES = {"door", "table", "wardrobe","chair"}
FOCUS_THRESHOLD = 0.45
GENERAL_THRESHOLD = 0.60

# for Stabilization settings
BUFFER_SIZE = 15
MIN_SUGGESTION_INTERVAL = 0.5
MIN_CONSECUTIVE_FRAMES = 8         
DOMINANT_THRESHOLD = 0.6            

# for Object-based stopping - Enhanced
STOP_OBJECTS = {"chair", "table", "person", "door", "wardrobe", "couch", "bed"}
DANGER_OBJECTS = {"person", "chair", "table"}  # Immediate danger objects
OBJECT_DANGER_THRESHOLD = 0.4
CENTRAL_ZONE_START = 0.35
CENTRAL_ZONE_END = 0.65

# for Enhanced detection settings
MAX_DETECTIONS = 10
