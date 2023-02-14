import cv2
import numpy as np
import zmq
import os
import tqdm
import matplotlib.pyplot as plt
import kornia as K
import kornia.feature as KF
import torch
from kornia_moons.feature import *
from core.core import Matcher


def to_torch_image(frame):
    img = K.image_to_tensor(frame).float() / 255.0
    img = K.color.bgr_to_rgb(img)
    # pad one dimension to make it a batch
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.cuda()
    return img


def get_matching_keypoints(lafs1, lafs2, idxs):
    xys2 = KF.get_laf_center(lafs1).squeeze()[idxs[:, 0]].detach().cpu().numpy()
    mkpts2 = KF.get_laf_center(lafs2).squeeze()[idxs[:, 1]].detach().cpu().numpy()
    return xys2, mkpts2


class LoFTRMatcher(Matcher):
    def __init__(self, cfg, device="cpu"):
        self.device = device
        self.dim_feature = cfg.dim_feature
        self.max_feature = cfg.max_feature
        self.thresh_confid = cfg.thresh_confid
        # dynamically config the matcher
        self.matcher = KF.LoFTR(pretrained=cfg.pretrained).eval()
        if self.device == "gpu":
            self.matcher = self.matcher.cuda()
        # loftr is detector-free method
        self.detector_free = True

    def match(self, image1, image2, xys1, xys2, desc1, desc2, score1, score2):
        # process image
        image1_tensor = to_torch_image(image1)
        image2_tensor = to_torch_image(image2)

        # matching with loftr
        input_dict = {
            "image0": K.color.rgb_to_grayscale(image1_tensor),
            "image1": K.color.rgb_to_grayscale(image2_tensor),
        }

        with torch.inference_mode():
            correspondences = self.matcher(input_dict)

        xys1 = correspondences["keypoints0"].cpu().numpy()
        xys2 = correspondences["keypoints1"].cpu().numpy()
        Fm, inliers = cv2.findFundamentalMat(
            xys1, xys2, cv2.USAC_MAGSAC, 0.5, 0.999, 100000
        )
        inliers = (inliers > 0).squeeze()
        xys1 = xys1[inliers]
        xys2 = xys2[inliers]
        # output of loftr is different from other matching algorithms
        return xys1, xys2, np.ones(xys1.shape[0]), None
