import numpy as np
import cv2
from tqdm import tqdm
import numpy as np
from core.core import Matcher
import matplotlib.pyplot as plt
import kornia as K
import kornia.feature as KF
import torch
from kornia_moons.feature import *


def to_torch_image(frame):
    img = K.image_to_tensor(frame).float() / 255.0
    img = K.color.bgr_to_rgb(img)
    # pad one dimension to make it a batch
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    img = img.cuda()
    return img


def get_matching_keypoints(image1, image2, idxs):
    xys1_matched = (
        KF.get_laf_center(image1).squeeze()[idxs[:, 0]].detach().cpu().numpy()
    )
    xys2_matched = (
        KF.get_laf_center(image2).squeeze()[idxs[:, 1]].detach().cpu().numpy()
    )
    return xys1_matched, xys2_matched


class AdalamMatcher(Matcher):
    def __init__(self, cfg, device="cpu"):
        self.device = device
        # FIXME: Adalam is actually not detector-free, but we use it as detector-free for now
        self.detector_free = True
        # self.dim_feature = cfg.dim_feature
        # self.max_feature = cfg.max_feature
        # self.thresh_confid = cfg.thresh_confid

    def match(self, image1, image2, xys1, xys2, desc1, desc2, score1, score2):
        # currently we can only use KFNet with adalam
        # TODO: integrate adalam with other detectors
        # process image
        image1_tensor = to_torch_image(image1)
        image2_tensor = to_torch_image(image2)

        # extract feature
        feature = KF.KeyNetAffNetHardNet(5000, True).eval().cuda()
        input_dict = {
            "image0": K.color.rgb_to_grayscale(
                image1_tensor
            ),  # LofTR works on grayscale images only
            "image1": K.color.rgb_to_grayscale(image2_tensor),
        }
        hw1 = torch.tensor(image1_tensor.shape[2:])
        hw2 = torch.tensor(image1_tensor.shape[2:])
        with torch.inference_mode():
            lafs1, resps1, descs1 = feature(K.color.rgb_to_grayscale(image1_tensor))
            lafs2, resps2, descs2 = feature(K.color.rgb_to_grayscale(image2_tensor))
            dists, idxs = KF.match_adalam(
                descs1.squeeze(0),
                descs2.squeeze(0),
                lafs1,
                lafs2,  # Adalam takes into account also geometric information
                config=None,
                hw1=hw1,
                hw2=hw2,
            )  # Adalam also benefits from knowing image size

        print(f"{idxs.shape[0]} tentative matches with AdaLAM")
        if idxs.shape[0] == 0:  # Early return
            return np.array([]), np.array([]), np.array([]), None

        # matching
        mkpts1, mkpts2 = get_matching_keypoints(lafs1, lafs2, idxs)
        Fm, inliers = cv2.findFundamentalMat(
            mkpts1, mkpts2, cv2.USAC_MAGSAC, 0.75, 0.999, 100000
        )
        inliers = inliers > 0
        dists = dists.cpu().numpy()
        return (
            mkpts1[inliers[:, 0]],
            mkpts2[inliers[:, 0]],
            dists[inliers[:, 0]],
            None,
        )
