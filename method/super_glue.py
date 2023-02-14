import numpy as np
import cv2
from tqdm import tqdm
import numpy as np
from core.core import Matcher

from pathlib import Path
from tqdm import tqdm
import matplotlib.cm as cm
import torch
from third_party.matching import Matching
from third_party.utils import frame2tensor


class SuperGlueMatcher(Matcher):
    def __init__(self, cfg, device="cpu"):
        self.device = device
        self.dim_feature = cfg.dim_feature
        self.max_feature = cfg.max_feature
        self.thresh_confid = cfg.thresh_confid
        # init superglue network
        if cfg.detector == "super_point":
            sp_config = {
                "superpoint": {
                    "nms_radius": cfg.nms_radius,
                    "keypoint_threshold": cfg.keypoint_threshold,
                    "max_keypoints": cfg.max_feature,
                },
                "superglue": {
                    "weights": cfg.superglue,
                    "sinkhorn_iterations": cfg.sinkhorn_iterations,
                    "match_threshold": cfg.match_threshold,
                },
            }
            self.matching = Matching(sp_config).eval()
            if self.device == "gpu":
                self.device_str = "cuda"
                self.matching = self.matching.cuda()
        else:
            raise NotImplementedError

    def match(self, image1, image2, xys1, xys2, desc1, desc2, score1, score2):
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        image1_tensor = frame2tensor(image1_gray, self.device_str)
        image2_tensor = frame2tensor(image2_gray, self.device_str)
        query_data = {
            "keypoints0": [torch.from_numpy(xys1[:, :2]).float().to(self.device_str)],
            "descriptors0": [torch.from_numpy(desc1.T).float().to(self.device_str)],
            "scores0": [torch.from_numpy(score1).float().to(self.device_str)],
            "image0": image1_tensor,
        }
        train_data = {
            "keypoints1": [torch.from_numpy(xys2[:, :2]).float().to(self.device_str)],
            "descriptors1": [torch.from_numpy(desc2.T).float().to(self.device_str)],
            "scores1": [torch.from_numpy(score2).float().to(self.device_str)],
            "image1": image2_tensor,
        }
        pred = self.matching({**query_data, **train_data})
        matches = pred["matches0"][0].cpu().numpy()
        confidence = pred["matching_scores0"][0].cpu().numpy()

        # create valid matches, -1 means invalid
        valid = matches > 0
        matched_idx_list = [[i, m] for i, m in enumerate(matches) if m > 0]
        matched_idx = np.array(matched_idx_list)
        confidence = confidence[valid]

        return xys1[matched_idx[:, 0]], xys2[matched_idx[:, 1]], confidence, None
