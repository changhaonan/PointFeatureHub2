import numpy as np
import cv2
from tqdm import tqdm

# add import path
import sys
import os

sys.path.append("../")
from core.core import Detector
from core.decorator import report_time
from tqdm import tqdm
from PIL import Image
import torch

from r2d2.tools import common
from r2d2.tools.dataloader import norm_RGB
from r2d2.nets.patchnet import *


def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint["net"])
    net = eval(checkpoint["net"])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint["state_dict"]
    net.load_state_dict({k.replace("module.", ""): v for k, v in weights.items()})
    return net.eval()


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = repeatability == self.max_filter(repeatability)

        # remove low peaks
        maxima *= repeatability >= self.rep_thr
        maxima *= reliability >= self.rel_thr

        return maxima.nonzero().t()[2:4]


def extract_multiscale(
    net,
    img,
    detector,
    scale_f=2**0.25,
    min_scale=0.0,
    max_scale=1,
    min_size=256,
    max_size=1024,
    verbose=False,
):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"

    assert max_scale <= 1
    s = 1.0  # current scale factor

    X, Y, S, C, Q, D = [], [], [], [], [], []
    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]
            if verbose:
                print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])

            # get output and reliability map
            descriptors = res["descriptors"][0]
            reliability = res["reliability"][0]
            repeatability = res["repeatability"][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y, x = detector(**res)  # nms
            c = reliability[0, 0, y, x]
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W / nw)
            Y.append(y.float() * H / nh)
            S.append((32 / s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode="bilinear", align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S)  # scale
    scores = torch.cat(C) * torch.cat(Q)  # scores = reliability * repeatability
    XYS = torch.stack([X, Y, S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores


class R2D2Detector(Detector):
    def __init__(self, cfg, device="cpu"):
        self.device = device
        self.dim_feature = cfg.dim_feature
        self.max_feature = cfg.max_feature
        self.thresh_confid = cfg.thresh_confid
        self.sift = cv2.xfeatures2d.SIFT_create()

        self.net = load_network(os.path.join(cfg.work_dir, cfg.model))
        self.detector = NonMaxSuppression(rel_thr=cfg.reliability_thr, rep_thr=cfg.repeatability_thr)
        if not self.device == "cpu":
            self.net = self.net.cuda()
        # parameters
        self.scale_f = cfg.scale_f
        self.min_scale = cfg.min_scale
        self.max_scale = cfg.max_scale
        self.min_size = cfg.min_size
        self.max_size = cfg.max_size

    @report_time
    def detect(self, image):
        # preprocess image
        image_tensor = norm_RGB(image)[None]
        if not self.device == "cpu":
            image_tensor = image_tensor.cuda()

        # extract keypoints
        xys, desc, scores = extract_multiscale(
            self.net,
            image_tensor,
            self.detector,
            scale_f=self.scale_f,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
            min_size=self.min_size,
            max_size=self.max_size,
            verbose=True,
        )
        xys = xys.cpu().numpy()
        desc = desc.cpu().numpy()
        scores = scores.cpu().numpy()
        idxs = scores.argsort()[-self.max_feature or None :]
        if len(idxs) == 0:
            return np.zeros((0, 2)), np.zeros((0, 32)), np.zeros((0,))
        return xys[idxs], desc[idxs], scores[idxs], image
