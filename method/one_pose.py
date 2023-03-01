import numpy as np
import cv2
import os
from tqdm import tqdm
import numpy as np
from core.core import Matcher32D
from core.decorator import report_time


class OnePoseMatcher(Matcher32D):
    def __init__(self, cfg, device="cpu"):
        self.device = device
        # sfm model
        self.avg_data = None
        self.clt_data = None
        self.kpts3d = None
        self.desc3d_avg = None
        self.desc3d_clt = None

    def load_sparse_model(self, model_path):
        avg_data_file = os.path.join(model_path, "anno", "anno_3d_average.npz")
        clt_data_file = os.path.join(model_path, "anno", "anno_3d_collect.npz")
        self.avg_data = np.load(avg_data_file)
        self.clt_data = np.load(clt_data_file)

    @report_time
    def match(self, image, kpts3d, kpts2d, desc3d_avg, desc3d_clt, desc2d, score2d):
        return kpts3d, kpts2d, np.array([]), None
