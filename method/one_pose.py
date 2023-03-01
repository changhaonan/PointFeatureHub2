import os
import cv2
import glob
import torch
import hydra
from tqdm import tqdm
import os.path as osp
import numpy as np

from PIL import Image
from loguru import logger
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from one_pose.utils import data_utils, path_utils, eval_utils, vis_utils

from pytorch_lightning import seed_everything

seed_everything(12345)

from core.core import Matcher32D
from core.decorator import report_time


def load_model(cfg):
    """Load model"""

    def load_matching_model(model_path):
        """Load onepose model"""
        from one_pose.models.GATsSPG_lightning_model import LitModelGATsSPG

        trained_model = LitModelGATsSPG.load_from_checkpoint(checkpoint_path=model_path)
        trained_model.cuda()
        trained_model.eval()
        trained_model.freeze()
        return trained_model

    def load_extractor_model(cfg, model_path):
        """Load extractor model(SuperPoint)"""
        from one_pose.models.extractors.SuperPoint.superpoint import SuperPoint
        from one_pose.sfm.extract_features import confs
        from one_pose.utils.model_io import load_network

        # currently only support superpoint
        extractor_model = SuperPoint(confs["superpoint"]["conf"])
        extractor_model.cuda()
        extractor_model.eval()
        load_network(extractor_model, model_path)

        return extractor_model

    matching_model = load_matching_model(cfg.match_model_path)
    extractor_model = load_extractor_model(cfg, cfg.extractor_model_path)
    return matching_model, extractor_model


def pack_data(avg_descriptors3d, clt_descriptors, keypoints3d, detection, image_size):
    """Prepare data for OnePose inference"""
    keypoints2d = torch.Tensor(detection["keypoints"])
    descriptors2d = torch.Tensor(detection["descriptors"])

    inp_data = {
        "keypoints2d": keypoints2d[None].cuda(),  # [1, n1, 2]
        "keypoints3d": keypoints3d[None].cuda(),  # [1, n2, 3]
        "descriptors2d_query": descriptors2d[None].cuda(),  # [1, dim, n1]
        "descriptors3d_db": avg_descriptors3d[None].cuda(),  # [1, dim, n2]
        "descriptors2d_db": clt_descriptors[None].cuda(),  # [1, dim, n2*num_leaf]
        "image_size": image_size,
    }
    return inp_data


class OnePoseMatcher(Matcher32D):
    def __init__(self, cfg, device="cpu"):
        self.device = device
        # parameters
        self.num_leaf = cfg.num_leaf
        # sfm model
        self.avg_data = None
        self.clt_data = None
        self.kpts3d = None
        self.desc3d_avg = None
        self.desc3d_clt = None
        # load matching model
        self.matching_model, self.extractor_model = load_model(cfg)

    def load_sparse_model(self, model_path):
        avg_data_file = os.path.join(model_path, "anno", "anno_3d_average.npz")
        clt_data_file = os.path.join(model_path, "anno", "anno_3d_collect.npz")
        self.avg_data = np.load(avg_data_file)
        self.clt_data = np.load(clt_data_file)
        self.idxs = np.load(os.path.join(model_path, "anno", "idxs.npy"))
        # parse data
        self.keypoints3d = torch.Tensor(self.clt_data["keypoints3d"]).cuda()
        self.num_3d = self.keypoints3d.shape[0]
        # Load average 3D features:
        self.avg_descriptors3d, _ = data_utils.pad_features3d_random(self.avg_data["descriptors3d"], self.avg_data["scores3d"], self.num_3d)
        # Load corresponding 2D features of each 3D point:
        self.clt_descriptors, _ = data_utils.build_features3d_leaves(
            self.clt_data["descriptors3d"], self.clt_data["scores3d"], self.idxs, self.num_3d, self.num_leaf
        )

    @report_time
    def match32d(self, image):
        # Normalize image:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        inp = transforms.ToTensor()(image_gray).cuda()[None]
        image_size = inp.shape[-2:]

        # Detect query image keypoints and extract descriptors:
        pred_detection = self.extractor_model(inp)
        pred_detection = {k: v[0].cpu().numpy() for k, v in pred_detection.items()}

        # 2D-3D matching by GATsSPG:
        inp_data = pack_data(self.avg_descriptors3d, self.clt_descriptors, self.keypoints3d, pred_detection, image_size)
        pred, _ = self.matching_model(inp_data)
        matches = pred["matches0"].detach().cpu().numpy()
        valid = matches > -1
        kpts2d = pred_detection["keypoints"]
        kpts3d = inp_data["keypoints3d"][0].detach().cpu().numpy()
        confidence = pred["matching_scores0"].detach().cpu().numpy()
        mkpts2d, mkpts3d, mconf = kpts2d[valid], kpts3d[matches[valid]], confidence[valid]

        return mkpts3d, mkpts2d, mconf, None
