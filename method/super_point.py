import numpy as np
from tqdm import tqdm
import numpy as np

# add import path
import sys
import cv2
from core.core import Detector
from core.decorator import report_time
from pathlib import Path
from tqdm import tqdm
import matplotlib.cm as cm
import torch
from third_party.superpoint import SuperPoint
from third_party.utils import frame2tensor

torch.set_grad_enabled(False)


class SuperPointDetector(Detector):
    def __init__(self, cfg, device="cpu"):
        self.device = device
        self.dim_feature = cfg.dim_feature
        self.max_feature = cfg.max_feature
        self.thresh_confid = cfg.thresh_confid
        # init superpoint network
        sp_config = {
            "nms_radius": cfg.nms_radius,
            "keypoint_threshold": cfg.keypoint_threshold,
            "max_keypoints": cfg.max_feature,
        }
        self.superpoint = SuperPoint(sp_config).eval()
        if self.device == "gpu":
            self.superpoint = self.superpoint.cuda()

    @report_time
    def detect(self, image):
        # preprocess image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if self.device == "gpu":
            frame_tensor = frame2tensor(image, "cuda")
        else:
            frame_tensor = frame2tensor(image, "cpu")
        result = self.superpoint({"image": frame_tensor})

        xys = result["keypoints"][0].cpu().numpy()
        # append a size channel to xys
        xys = np.concatenate((xys, np.ones((xys.shape[0], 1))), axis=1)
        desc = result["descriptors"][0].cpu().numpy().T  # Transpose to be compatible with ORB
        scores = result["scores"][0].cpu().numpy()
        idxs = scores.argsort()[-self.max_feature or None :]

        if len(idxs) == 0:
            return np.zeros((0, 2)), np.zeros((0, 32)), np.zeros((0,)), image
        return xys[idxs], desc[idxs], scores[idxs], image
