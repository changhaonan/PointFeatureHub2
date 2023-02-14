import hydra
import os
import cv2
import glob
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from method import detector_map, matcher_map
from core.wrapper import (
    DrawKeyPointsDetectorWrapper,
    SaveImageDetectorWrapper,
    NetworkDetectorWrapper,
    DrawKeyPointsMatcherWrapper,
    SaveImageMatcherWrapper,
    NetworkMatcherWrapper,
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

    # register pillow_heif to read HEIC images
    register_heif_opener()

    if cfg.task == "detect":
        detector = create_detector_thunk()
        # go over train list
        for image_file in glob.glob(os.path.join(cfg.data_dir, cfg.train_dir, "*.png")):
            image = cv2.imread(image_file)
            detector.detect(image)
    elif cfg.task == "match":
        matcher = create_matcher_thunk()
        if not matcher.detector_free:
            detector = create_detector_thunk()
        else:
            detector = None

        # go over train list
        image_prev, xys_prev, desc_prev, scores_prev = None, None, None, None
        for image_train_file in glob.glob(
            os.path.join(cfg.data_dir, cfg.train_dir, "*")
        ):
            # get image name
            image_name = os.path.basename(image_train_file)
            image_query_file = os.path.join(cfg.data_dir, cfg.query_dir, image_name)

            if not image_name.endswith(".HEIC"):
                image_train = cv2.imread(image_train_file)
                image_query = cv2.imread(image_query_file)
            else:
                # use pillow_heif to read HEIC images
                image_train_pil = Image.open(
                    image_train_file
                )  # do whatever need with a Pillow image
                image_query_pil = Image.open(image_query_file)
                image_train = cv2.cvtColor(np.array(image_train_pil), cv2.COLOR_BGR2RGB)
                image_query = cv2.cvtColor(np.array(image_query_pil), cv2.COLOR_BGR2RGB)

            # resize image based on max_height and max_width
            if (
                image_train.shape[0] > cfg.max_height
                or image_train.shape[1] > cfg.max_width
            ):
                scale = min(
                    cfg.max_height / image_train.shape[0],
                    cfg.max_width / image_train.shape[1],
                )
                image_train = cv2.resize(image_train, (0, 0), fx=scale, fy=scale)
            if (
                image_query.shape[0] > cfg.max_height
                or image_query.shape[1] > cfg.max_width
            ):
                scale = min(
                    cfg.max_height / image_query.shape[0],
                    cfg.max_width / image_query.shape[1],
                )
                image_query = cv2.resize(image_query, (0, 0), fx=scale, fy=scale)

            if not matcher.detector_free:
                xys_train, desc_train, scores_train, _ = detector.detect(image_train)
                xys_query, desc_query, scores_query, _ = detector.detect(image_query)
            else:
                xys_train, desc_train, scores_train = None, None, None
                xys_query, desc_query, scores_query = None, None, None

            matcher.match(
                image_query,
                image_train,
                xys_query,
                xys_train,
                desc_query,
                desc_train,
                scores_query,
                scores_train,
            )


if __name__ == "__main__":
    launch_detector_hydra()
