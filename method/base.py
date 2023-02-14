from core.core import Loader, Matcher, Detector
import numpy as np


class BaseDetector(Detector):
    def __init__(self, cfg):
        self.cfg = cfg

    def detect(self, image):
        return np.array([]), np.array([]), np.array([]), np.array([])


class BaseMatcher(Matcher):
    def __init__(self, cfg):
        self.cfg = cfg

    def match(self, image1, image2, xys1, xys2, desc1, desc2, score1, score2):
        return np.array([]), np.array([]), np.array([]), np.array([])


class BaseLoder(Loader):
    def __init__(self, cfg):
        self.cfg = cfg

    def load(self, image1_name, image2_name):
        return np.array([]), np.array([])
