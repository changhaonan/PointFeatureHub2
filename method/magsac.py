import numpy as np
import cv2
from tqdm import tqdm
import numpy as np
from core.core import Matcher
from core.decorator import report_time


class MAGSACMatcher(Matcher):
    def __init__(self, cfg, device="cpu"):
        self.device = device
        self.dim_feature = cfg.dim_feature
        self.max_feature = cfg.max_feature
        self.thresh_confid = cfg.thresh_confid
        self.min_prior_matches = cfg.min_prior_matches
        # dynamically config the matcher
        if cfg.detector == "orb":
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif cfg.detector == "super_point" or cfg.detector == "sift" or cfg.detector == "surf" or cfg.detector == "r2d2":
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            raise NotImplementedError

    @report_time
    def match(self, image1, image2, xys1, xys2, desc1, desc2, score1, score2):
        # do a match first
        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        matched_idx = np.array([[m.queryIdx, m.trainIdx] for m in matches])
        confidence = np.array([m.distance for m in matches])
        xys1_prior_matches = xys1[matched_idx[:, 0]]
        xys2_prior_matches = xys2[matched_idx[:, 1]]
        if matched_idx.shape[0] <= self.min_prior_matches:
            return xys1, xys2, np.array([]), None

        Fm, inliers = cv2.findFundamentalMat(xys1_prior_matches[:, :2], xys2_prior_matches[:, :2], cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
        inliers = (inliers > 0).squeeze()
        xys1 = xys1_prior_matches[inliers]
        xys2 = xys2_prior_matches[inliers]
        confidence = confidence[inliers]
        return xys1, xys2, confidence, None
