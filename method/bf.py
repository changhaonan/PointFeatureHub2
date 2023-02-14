import numpy as np
import cv2
from tqdm import tqdm
import numpy as np
from core.core import Matcher


class BFMatcher(Matcher):
    def __init__(self, cfg, device="cpu"):
        self.device = device
        self.dim_feature = cfg.dim_feature
        self.max_feature = cfg.max_feature
        self.thresh_confid = cfg.thresh_confid
        # dynamically config the matcher
        if cfg.detector == "orb":
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif (
            cfg.detector == "super_point"
            or cfg.detector == "sift"
            or cfg.detector == "surf"
            or cfg.detector == "r2d2"
        ):
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            raise NotImplementedError

    def match(self, image1, image2, xys1, xys2, desc1, desc2, score1, score2):
        # match two sets of descriptors
        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        matched_idx = np.array([[m.queryIdx, m.trainIdx] for m in matches])
        confidence = np.array([m.distance for m in matches])
        return xys1[matched_idx[:, 0]], xys2[matched_idx[:, 1]], confidence, None
