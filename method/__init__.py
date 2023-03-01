from .sift import SIFTDetector
from .super_point import SuperPointDetector
from .r2d2 import R2D2Detector
from .orb import ORB2Detector

from .bf import BFMatcher
from .magsac import MAGSACMatcher
from .super_glue import SuperGlueMatcher
from .loftr import LoFTRMatcher
from .adalam import AdalamMatcher

from .one_pose import OnePoseMatcher

from .base import BaseDetector, BaseDetector, BaseLoder

detector_map = {
    "base": BaseDetector,
    "sift": SIFTDetector,
    "super_point": SuperPointDetector,
    "r2d2": R2D2Detector,
    "orb": ORB2Detector,
}

matcher_map = {
    "base": BaseDetector,
    "bf": BFMatcher,
    "magsac": MAGSACMatcher,
    "super_glue": SuperGlueMatcher,
    "loftr": LoFTRMatcher,
    "adalam": AdalamMatcher,
}

matcher32D_map = {
    "one_pose": OnePoseMatcher,
}

loader_map = {
    "base": BaseLoder,
}
