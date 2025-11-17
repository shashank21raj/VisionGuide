import pathlib
from ultralytics import YOLO
from config import DEVICE

def loadModel():
    print("Loading YOLO models...")
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    main_model = YOLO("Custom_yolo11n.pt")
    focus_model = YOLO("Focused_yolo11n.pt")
    main_model.to(DEVICE)
    focus_model.to(DEVICE)
    pathlib.PosixPath = temp
    print("All models loaded!")
    return main_model, focus_model