import hydra
import os
import cv2
import glob
import numpy as np
from PIL import Image

from method import detector_map, matcher_map, loader_map
from core.wrapper import (
    DrawKeyPointsDetectorWrapper,
    SaveImageDetectorWrapper,
    NetworkDetectorWrapper,
    DrawKeyPointsMatcherWrapper,
    SaveImageMatcherWrapper,
    NetworkMatcherWrapper,
    FileLoaderWrapper,
    NetworkLoaderWrapper,
)


@hydra.main(config_path="cfg", config_name="config")
def launch_detector_hydra(cfg):
    def create_detector_thunk(**kwargs):
        if cfg.detector not in detector_map:
            raise ValueError(
                "Detector {} not supported. Supported detectors are: {}".format(
                    cfg.detector, detector_map.keys()
                )
            )
        detector = detector_map[cfg.detector](cfg, cfg.detector_device, **kwargs)
        if cfg.draw_keypoints:
            window_name = f"{cfg.task}:{cfg.detector}"
            detector = DrawKeyPointsDetectorWrapper(detector, window_name=window_name)
            if cfg.save_image:
                detector = SaveImageDetectorWrapper(
                    detector,
                    cfg.save_dir,
                    prefix=cfg.prefix,
                    suffix=cfg.suffix,
                    padding_zeros=cfg.padding_zeros,
                    verbose=cfg.verbose,
                )
        if cfg.publish_detector:
            detector = NetworkDetectorWrapper(detector, cfg.detector_port)
        return detector

    def create_matcher_thunk(**kwargs):
        if cfg.matcher not in matcher_map:
            raise ValueError(
                "Matcher {} not supported. Supported matchers are: {}".format(
                    cfg.matcher, matcher_map.keys()
                )
            )
        matcher = matcher_map[cfg.matcher](cfg, cfg.matcher_device, **kwargs)
        if cfg.draw_matches:
            window_name = f"{cfg.task}:{cfg.detector}+{cfg.matcher}"
            matcher = DrawKeyPointsMatcherWrapper(matcher, window_name=window_name)
            if cfg.save_image:
                matcher = SaveImageMatcherWrapper(
                    matcher,
                    cfg.save_dir,
                    prefix=cfg.prefix,
                    suffix=cfg.suffix,
                    padding_zeros=cfg.padding_zeros,
                    verbose=cfg.verbose,
                )
        if cfg.publish_matcher:
            matcher = NetworkMatcherWrapper(matcher, cfg.matcher_port)
        return matcher

    def create_loader_thunk(**kwargs):
        if cfg.loader not in loader_map:
            raise ValueError(
                "Matcher {} not supported. Supported matchers are: {}".format(
                    cfg.matcher, loader_map.keys()
                )
            )
        loader = loader_map[cfg.matcher](cfg, **kwargs)
        if cfg.load_from_network:
            loader = NetworkMatcherWrapper(loader, cfg.loader_port)
        else:
            loader = FileLoaderWrapper(
                loader,
                os.path.join(cfg.data_dir, cfg.train_dir),
                os.path.join(cfg.data_dir, cfg.query_dir),
            )
        return loader

    # create loader
    loader = create_loader_thunk()

    if cfg.task == "detect":
        detector = create_detector_thunk()
        # go over train list
        for image_file in glob.glob(os.path.join(cfg.data_dir, cfg.train_dir, "*.png")):
            image = cv2.imread(image_file)
            detector.detect(image)
    elif cfg.task == "match":
        matcher = create_matcher_thunk()
        detector = create_detector_thunk()

        for image1_file in glob.glob(os.path.join(cfg.data_dir, cfg.train_dir, "*")):
            # get image name
            image_name = os.path.basename(image1_file)
            image1, image2 = loader.load(image_name, image_name)

            # resize image based on max_height and max_width
            if image1.shape[0] > cfg.max_height or image1.shape[1] > cfg.max_width:
                scale = min(
                    cfg.max_height / image1.shape[0],
                    cfg.max_width / image1.shape[1],
                )
                image1 = cv2.resize(image1, (0, 0), fx=scale, fy=scale)
            if image2.shape[0] > cfg.max_height or image2.shape[1] > cfg.max_width:
                scale = min(
                    cfg.max_height / image2.shape[0],
                    cfg.max_width / image2.shape[1],
                )
                image2 = cv2.resize(image2, (0, 0), fx=scale, fy=scale)

            xys1, desc1, scores1, _ = detector.detect(image1)
            xys2, desc2, scores2, _ = detector.detect(image2)

            matcher.match(
                image1,
                image2,
                xys1,
                xys2,
                desc1,
                desc2,
                scores1,
                scores2,
            )


if __name__ == "__main__":
    launch_detector_hydra()
