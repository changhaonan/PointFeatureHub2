import hydra
import os
import cv2
import glob
import zmq
import numpy as np
from method import detector_map, matcher_map, loader_map, matcher32D_map
from core.wrapper import (
    DrawKeyPointsDetectorWrapper,
    SaveImageDetectorWrapper,
    NetworkDetectorWrapper,
    DrawKeyPointsMatcherWrapper,
    SaveImageMatcherWrapper,
    NetworkMatcherWrapper,
    FileLoaderWrapper,
    NetworkLoaderWrapper,
    NetworkMatcher32DWrapper,
    DrawKeyPointsMatcher32DWrapper,
)


@hydra.main(config_path="cfg", config_name="config")
def launch_detector_hydra(cfg):
    def create_detector_thunk(**kwargs):
        if cfg.detector not in detector_map:
            raise ValueError("Detector {} not supported. Supported detectors are: {}".format(cfg.detector, detector_map.keys()))
        detector = detector_map[cfg.detector](cfg, cfg.detector_device)
        if cfg.draw_keypoints:
            window_name = f"{cfg.task}:{cfg.detector}"
            detector = DrawKeyPointsDetectorWrapper(detector, window_name=window_name)
            if cfg.save_image:
                detector = SaveImageDetectorWrapper(
                    detector, cfg.save_dir, prefix=cfg.prefix, suffix=cfg.suffix, padding_zeros=cfg.padding_zeros, verbose=cfg.verbose
                )
        if kwargs["publish_to_network"]:
            detector = NetworkDetectorWrapper(detector, kwargs["context"], kwargs["socket"])
        return detector

    def create_matcher_thunk(**kwargs):
        if cfg.matcher not in matcher_map:
            raise ValueError("Matcher {} not supported. Supported matchers are: {}".format(cfg.matcher, matcher_map.keys()))
        matcher = matcher_map[cfg.matcher](cfg, cfg.matcher_device)
        if cfg.draw_matches:
            window_name = f"{cfg.task}:{cfg.detector}+{cfg.matcher}"
            matcher = DrawKeyPointsMatcherWrapper(matcher, window_name=window_name)
            if cfg.save_image:
                matcher = SaveImageMatcherWrapper(
                    matcher, cfg.save_dir, prefix=cfg.prefix, suffix=cfg.suffix, padding_zeros=cfg.padding_zeros, verbose=cfg.verbose
                )
        if kwargs["publish_to_network"]:
            matcher = NetworkMatcherWrapper(matcher, kwargs["context"], kwargs["socket"])
        return matcher

    def create_matcher32D_thunk(**kwargs):
        if cfg.matcher not in matcher32D_map:
            raise ValueError("Matcher {} not supported. Supported matchers are: {}".format(cfg.matcher, matcher_map.keys()))
        matcher = matcher32D_map[cfg.matcher](cfg, cfg.matcher_device)
        if cfg.draw_matches:
            window_name = f"{cfg.task}:{cfg.detector}+{cfg.matcher}"
            matcher = DrawKeyPointsMatcher32DWrapper(matcher, window_name=window_name)
            # if cfg.save_image:
            #     matcher = SaveImageMatcherWrapper(
            #         matcher, cfg.save_dir, prefix=cfg.prefix, suffix=cfg.suffix, padding_zeros=cfg.padding_zeros, verbose=cfg.verbose
            #     )
        if kwargs["publish_to_network"]:
            matcher = NetworkMatcher32DWrapper(matcher, kwargs["context"], kwargs["socket"])
        return matcher

    def create_loader_thunk(**kwargs):
        if cfg.loader not in loader_map:
            raise ValueError("Matcher {} not supported. Supported matchers are: {}".format(cfg.matcher, loader_map.keys()))
        loader = loader_map[cfg.loader](cfg)
        if cfg.load_from_network:
            loader = NetworkLoaderWrapper(loader, kwargs["context"], kwargs["socket"])
        else:
            loader = FileLoaderWrapper(loader, os.path.join(cfg.data_dir, cfg.train_dir), os.path.join(cfg.data_dir, cfg.query_dir))
        return loader

    # create loader
    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.REP)
    zmq_socket.bind(f"tcp://0.0.0.0:{cfg.port}")
    loader = create_loader_thunk(context=zmq_context, socket=zmq_socket)

    if cfg.task == "detect":
        publish_to_network = cfg.publish_to_network
        detector = create_detector_thunk(context=zmq_context, socket=zmq_socket, publish_to_network=publish_to_network)
        # go over train list
        if not cfg.load_from_network:
            for image_file in glob.glob(os.path.join(cfg.data_dir, cfg.train_dir, "*.png")):
                image = cv2.imread(image_file)
                detector.detect(image)
        else:
            while True:
                image1, image2 = loader.load("network", "none")
                detector.detect(image1)
    elif cfg.task == "match":
        publish_to_network = cfg.publish_to_network
        matcher = create_matcher_thunk(context=zmq_context, socket=zmq_socket, publish_to_network=publish_to_network)
        detector = create_detector_thunk(context=zmq_context, socket=zmq_socket, publish_to_network=False)
        if not cfg.load_from_network:
            for image1_file in glob.glob(os.path.join(cfg.data_dir, cfg.train_dir, "*")):
                # get image name
                image_name = os.path.basename(image1_file)
                image1, image2 = loader.load(image_name, image_name)

                # resize image based on max_height and max_width
                if image1.shape[0] > cfg.max_height or image1.shape[1] > cfg.max_width:
                    scale = min(cfg.max_height / image1.shape[0], cfg.max_width / image1.shape[1])
                    image1 = cv2.resize(image1, (0, 0), fx=scale, fy=scale)
                if image2.shape[0] > cfg.max_height or image2.shape[1] > cfg.max_width:
                    scale = min(cfg.max_height / image2.shape[0], cfg.max_width / image2.shape[1])
                    image2 = cv2.resize(image2, (0, 0), fx=scale, fy=scale)

                xys1, desc1, scores1, _ = detector.detect(image1)
                xys2, desc2, scores2, _ = detector.detect(image2)

                matcher.match(image1, image2, xys1, xys2, desc1, desc2, scores1, scores2)
        else:
            while True:
                image1, image2 = loader.load("network", "network")
                xys1, desc1, scores1, _ = detector.detect(image1)
                xys2, desc2, scores2, _ = detector.detect(image2)
                matcher.match(image1, image2, xys1, xys2, desc1, desc2, scores1, scores2)
    elif cfg.task == "match32D":
        publish_to_network = cfg.publish_to_network
        matcher = create_matcher32D_thunk(context=zmq_context, socket=zmq_socket, publish_to_network=publish_to_network)
        if not cfg.load_from_network:
            for scene_path in glob.glob(os.path.join(cfg.data_dir, cfg.train_dir, "*")):
                # load sparse model
                matcher.load_sparse_model(scene_path)
                # get image name
                scene_name = os.path.basename(scene_path)
                # load intrinsics
                query_dir = os.path.join(cfg.data_dir, cfg.query_dir, scene_name)
                K = np.loadtxt(os.path.join(query_dir, "intrinsics.txt"))
                for image_file in glob.glob(os.path.join(query_dir, "*")):
                    if image_file.endswith(".txt"):
                        continue
                    image = cv2.imread(image_file)
                    matcher.match32d(image, K)


if __name__ == "__main__":
    launch_detector_hydra()
