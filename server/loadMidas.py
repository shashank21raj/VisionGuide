import torch
from config import DEVICE



def loadMidas():
    print("Loading MiDaS_small...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(DEVICE)
    midas.eval()
    midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    return midas, midas_transform