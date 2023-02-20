import numpy as np
import cv2
from tqdm import tqdm
import numpy as np

# add import path
import sys

sys.path.append("../")
from core.core import Detector
from core.decorator import report_time


class SIFTDetector(Detector):
    def __init__(self, cfg, device="cpu"):
        self.device = device
        self.dim_feature = cfg.dim_feature
        self.max_feature = cfg.max_feature
        self.thresh_confid = cfg.thresh_confid
        self.sift = cv2.xfeatures2d.SIFT_create()

    @report_time
    def detect(self, image):
        # detect keypoints/descriptors for a single image
        kpts, desc = self.sift.detectAndCompute(image, None)
        xys = np.array([[k.pt[0], k.pt[1], k.size] for k in kpts])
        scores = np.array([k.response for k in kpts])
        idxs = scores.argsort()[-self.max_feature or None :]
        if len(idxs) == 0:
            return np.zeros((0, 2)), np.zeros((0, 32)), np.zeros((0,))
        return xys[idxs], desc[idxs], scores[idxs], image
