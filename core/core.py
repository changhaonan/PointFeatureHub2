from typing import Dict, Any, Tuple
import abc
from abc import ABC
from abc import abstractmethod
import numpy as np


class Detector(ABC):
    """Abstract class for detector."""

    # Set this in SOME subclasses
    metadata = {}

    # Set this in ALL subclasses
    device = None
    dim_feature = None  # dimension of the feature
    max_feature = None  # maximum number of features in an image
    thresh_confid = None  # threshold of the feature confidence

    @abc.abstractmethod
    def detect(self, image) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Detect keypoints and descriptors from a single image.
        Args:
            image (np.ndarray): image to be detected.
        Returns:
            xys (np.ndarray): keypoints' coordinates and size.
            desc (np.ndarray): descriptors.
            scores (np.ndarray): scores of keypoints.
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            Detector: The base Detector instance
        """
        return self


class DetectorWrapper(Detector):
    def __init__(self, detector):
        self.detector = detector

        self._dim_feature = None
        self._max_feature = None
        self._thresh_confid = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.env, name)

    @property
    def dim_feature(self):
        if self._dim_feature is None:
            return self.detector.dim_feature
        return self._dim_feature

    @dim_feature.setter
    def dim_feature(self, value):
        self._dim_feature = value

    @property
    def max_feature(self):
        if self._max_feature is None:
            return self.detector.max_feature
        return self._max_feature

    @max_feature.setter
    def max_feature(self, value):
        self._max_feature = value

    @property
    def thresh_confid(self):
        if self._thresh_confid is None:
            return self.detector.thresh_confid
        return self._thresh_confid

    @thresh_confid.setter
    def thresh_confid(self, value):
        self._thresh_confid = value

    def detect(self, image):
        return self.detector.detect(image)

    @property
    def unwrapped(self):
        return self.detector.unwrapped


class Matcher(ABC):
    """Abstract class for matcher."""

    # Set this in SOME subclasses
    metadata = {}

    # Set this in ALL subclasses
    device = None
    dim_feature = None  # dimension of the feature
    max_feature = None  # maximum number of features in an image
    thresh_confid = None  # threshold of the feature confidence
    detector_free = False  # whether the matcher is detector-free

    @abc.abstractmethod
    def match(
        self, image1, image2, xys1, xys2, desc1, desc2, score1, score2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Match keypoints and descriptors from two images.
        Args:
            image1 (np.ndarray): image1 to be matched.
            image2 (np.ndarray): image2 to be matched.
            xys1 (np.ndarray): keypoints' coordinates and size of image1.
            xys2 (np.ndarray): keypoints' coordinates and size of image2.
            desc1 (np.ndarray): descriptors of image1.
            desc2 (np.ndarray): descriptors of image2.
            scores1 (np.ndarray): scores of keypoints of image1.
            scores2 (np.ndarray): scores of keypoints of image2.
        Returns:
            xys1_matched (np.ndarray): matched keypoints' coordinates and size of image1.
            xys2_matched (np.ndarray): matched keypoints' coordinates and size of image2.
            confid_matched (np.ndarray): confidence of the matches.
            vis_image (np.ndarray): visualization image.
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base Matcher instance
        """
        return self


class MatcherWrapper(Matcher):
    def __init__(self, matcher):
        self.matcher = matcher

        self._dim_feature = None
        self._max_feature = None
        self._thresh_confid = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.env, name)

    @property
    def dim_feature(self):
        if self._dim_feature is None:
            return self.matcher.dim_feature
        return self._dim_feature

    @dim_feature.setter
    def dim_feature(self, value):
        self._dim_feature = value

    @property
    def max_feature(self):
        if self._max_feature is None:
            return self.matcher.max_feature
        return self._max_feature

    @max_feature.setter
    def max_feature(self, value):
        self._max_feature = value

    @property
    def thresh_confid(self):
        if self._thresh_confid is None:
            return self.matcher.thresh_confid
        return self._thresh_confid

    @thresh_confid.setter
    def thresh_confid(self, value):
        self._thresh_confid = value

    def match(self, image1, image2, xys1, xys2, desc1, desc2, score1, score2):
        return self.matcher.match(
            image1, image2, xys1, xys2, desc1, desc2, score1, score2
        )

    @property
    def detector_free(self):
        return self.matcher.detector_free

    @detector_free.setter
    def detector_free(self, value):
        self.matcher.detector_free = value

    @property
    def unwrapped(self):
        return self.matcher.unwrapped
