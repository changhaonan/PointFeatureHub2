import cv2
import os
from core.core import DetectorWrapper, MatcherWrapper, LoaderWrapper

import numpy as np
import matplotlib.cm as cm
from third_party.utils import make_matching_plot_fast
import zmq
from pillow_heif import register_heif_opener
from PIL import Image


class SaveImageDetectorWrapper(DetectorWrapper):
    """Save detected image to a file"""

    def __init__(
        self,
        detector,
        save_dir: str,
        prefix: str = "image",
        suffix: str = "png",
        padding_zeros: int = 4,
        verbose: bool = False,
    ):
        super(SaveImageDetectorWrapper, self).__init__(detector)
        self.save_dir = save_dir
        self.prefix = prefix
        self.suffix = suffix
        self.verbose = verbose
        self.padding_zeros = padding_zeros

        # clean up the save_dir
        if os.path.exists(self.save_dir):
            os.system("rm -rf {}".format(self.save_dir))
        os.makedirs(self.save_dir)
        # counter for image
        self.counter = 0

    def detect(self, image):
        # detect keypoints/descriptors for a single image
        xys, desc, scores, vis_image = self.detector.detect(image)
        # save image
        self.save(image)
        return xys, desc, scores, vis_image

    def save(self, image):
        # save image
        filename = os.path.join(
            self.save_dir,
            "{}_{}.{}".format(
                self.prefix, str(self.counter).zfill(self.padding_zeros), self.suffix
            ),
        )
        cv2.imwrite(filename, image)
        if self.verbose:
            print("Save image to {}".format(filename))
        self.counter += 1


class DrawKeyPointsDetectorWrapper(DetectorWrapper):
    """Draw keypoints on image and visualize it"""

    def __init__(self, detector, window_name: str = "image", vis_height=500, show=True):
        super(DrawKeyPointsDetectorWrapper, self).__init__(detector)
        self.window_name = window_name
        self.vis_height = vis_height
        self.show = show

    def detect(self, image):
        # detect keypoints/descriptors for a single image
        xys, desc, scores, vis_image = self.detector.detect(image)
        # visualize image
        vis_image = self.vis(image, xys, scores)
        return xys, desc, scores, vis_image

    def vis(self, image, xys, scores):
        vis_image = image.copy()
        # resize image height to 500
        scale = self.vis_height / vis_image.shape[0]
        vis_image = cv2.resize(vis_image, None, fx=scale, fy=scale)
        # draw keypoints with colormap using scores
        vis_image = cv2.drawKeypoints(
            vis_image,
            [cv2.KeyPoint(x * scale, y * scale, 1) for x, y, s in xys],
            vis_image,
            flags=0,
        )
        # visualize image
        if self.show:
            cv2.imshow(self.window_name, vis_image)
            cv2.waitKey(0)
        return vis_image


class NetworkDetectorWrapper(DetectorWrapper):
    """Send result to web socket."""

    def __init__(self, detector, context, socket):
        super().__init__(detector)
        self.context = context
        self.socket = socket

    def detect(self, image):
        # detect keypoints/descriptors for a single image
        xys, desc, scores, vis_image = self.detector.detect(image)
        # send result to web socket
        num_feat, feat_dim = desc.shape
        msg = np.array([num_feat, feat_dim]).reshape(-1).astype(np.int32).tobytes()
        self.socket.send(msg, 2)
        msg = xys.astype(np.float32).reshape(-1).tobytes()
        self.socket.send(msg, 2)
        msg = desc.astype(np.float32).reshape(-1).tobytes()
        self.socket.send(msg, 0)
        return xys, desc, scores, vis_image


class SaveImageMatcherWrapper(MatcherWrapper):
    """Save detected image to a file"""

    def __init__(
        self,
        matcher,
        save_dir: str,
        prefix: str = "image",
        suffix: str = "png",
        padding_zeros: int = 4,
        verbose: bool = False,
    ):
        super(SaveImageMatcherWrapper, self).__init__(matcher)
        self.save_dir = save_dir
        self.prefix = prefix
        self.suffix = suffix
        self.verbose = verbose
        self.padding_zeros = padding_zeros

        # clean up the save_dir
        if os.path.exists(self.save_dir):
            os.system("rm -rf {}".format(self.save_dir))
        os.makedirs(self.save_dir)
        # counter for image
        self.counter = 0

    def match(self, image1, image2, xys1, xys2, desc1, desc2, score1, score2):
        # do match
        xys1_matched, xys2_matched, confidence, vis_image = self.matcher.match(
            image1, image2, xys1, xys2, desc1, desc2, score1, score2
        )
        # save image
        self.save(vis_image)
        return xys1_matched, xys2_matched, confidence, vis_image

    def save(self, image):
        # save image
        filename = os.path.join(
            self.save_dir,
            "{}_{}.{}".format(
                self.prefix, str(self.counter).zfill(self.padding_zeros), self.suffix
            ),
        )
        cv2.imwrite(filename, image)
        if self.verbose:
            print("Save image to {}".format(filename))
        self.counter += 1


class DrawKeyPointsMatcherWrapper(MatcherWrapper):
    """Draw keypoints & matching lines on image and visualize it"""

    def __init__(self, matcher, window_name: str = "image", vis_height=500, show=True):
        super(DrawKeyPointsMatcherWrapper, self).__init__(matcher)
        self.window_name = window_name
        self.vis_height = vis_height
        self.show = show

    def match(self, image1, image2, xys1, xys2, desc1, desc2, score1, score2):
        xys1_matched, xys2_matched, confidence, _ = self.matcher.match(
            image1, image2, xys1, xys2, desc1, desc2, score1, score2
        )

        if self.detector_free:
            xys1 = xys1_matched
            xys2 = xys2_matched
        # visualize image
        vis_image = self.vis(
            image1, image2, xys1, xys2, xys1_matched, xys2_matched, confidence
        )
        return xys1_matched, xys2_matched, confidence, vis_image

    def vis(self, image1, image2, xys1, xys2, xys1_matched, xys2_matched, confidence):
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        if np.std(confidence) < 1.0:
            # if the confidence is similar, use green color
            color_green = np.array([0.0, 1.0, 0.0])[None, :]
            color = np.repeat(color_green, confidence.shape[0], axis=0)
        else:
            color = cm.hot(confidence)
        text = [
            "Keypoints: {}:{}".format(len(xys1), len(xys2)),
            "Matches: {}".format(len(xys1_matched)),
        ]

        if xys1_matched.shape[0] == 0:
            color = np.array([[0.0, 0.0, 0.0]])
            text = ["No matches found"]
            vis_image = make_matching_plot_fast(
                image1_gray,
                image2_gray,
                xys1,
                xys2,
                xys1_matched,
                xys2_matched,
                color,
                text,
                path=None,
                show_keypoints=True,
            )
        else:
            # visualize matches
            vis_image = make_matching_plot_fast(
                image1_gray,
                image2_gray,
                xys1[:, :2],
                xys2[:, :2],
                xys1_matched[:, :2],
                xys2_matched[:, :2],
                color,
                text,
                path=None,
                show_keypoints=True,
            )

        if self.show:
            cv2.imshow(self.window_name, vis_image)
            cv2.waitKey(0)
        return vis_image


class NetworkMatcherWrapper(MatcherWrapper):
    """Send result to web socket."""

    def __init__(self, matcher, context, socket):
        super().__init__(matcher)
        self.context = context
        self.socket = socket

    def match(self, image1, image2, xys1, xys2, desc1, desc2, score1, score2):
        xys1_matched, xys2_matched, confidence, vis_image = self.matcher.match(
            image1, image2, xys1, xys2, desc1, desc2, score1, score2
        )
        # send result to web socket
        num_matched = xys1_matched.shape[0]
        msg = np.array([num_matched]).astype(np.int32).tobytes()
        self.socket.send(msg, 2)
        msg = xys1_matched.astype(np.float32).reshape(-1).tobytes()
        self.socket.send(msg, 2)
        msg = xys2_matched.astype(np.float32).reshape(-1).tobytes()
        self.socket.send(msg, 2)
        msg = confidence.astype(np.float32).reshape(-1).tobytes()
        self.socket.send(msg, 0)
        return xys1_matched, xys2_matched, confidence, vis_image


class FileLoaderWrapper(LoaderWrapper):
    """Data load by reading from file"""

    def __init__(self, loader, image_train_dir, image_query_dir):
        super().__init__(loader)
        self.image_train_dir = image_train_dir
        self.image_query_dir = image_query_dir
        # register pillow_heif to read HEIC images
        register_heif_opener()

    def load(self, image1_name, image2_name):
        # load from load
        image1, image2 = self.loader.load(image1_name, image2_name)

        # overwrite with file
        image1_file = os.path.join(self.image_train_dir, image1_name)
        image2_file = os.path.join(self.image_query_dir, image2_name)

        if os.path.exists(image1_file):
            if not image1_name.endswith(".HEIC"):
                image1 = cv2.imread(image1_file)
            else:
                # use pillow_heif to read HEIC images
                image1_pil = Image.open(image1_file)
                image1 = cv2.cvtColor(np.array(image1_pil), cv2.COLOR_BGR2RGB)
        if os.path.exists(image2_file):
            if not image2_name.endswith(".HEIC"):
                image2 = cv2.imread(image2_file)
            else:
                # use pillow_heif to read HEIC images
                image2_pil = Image.open(image2_file)
                image2 = cv2.cvtColor(np.array(image2_pil), cv2.COLOR_BGR2RGB)
        return image1, image2


class NetworkLoaderWrapper(LoaderWrapper):
    """Data load by reading from web socket."""

    def __init__(self, loader, context, socket):
        super().__init__(loader)
        self.context = context
        self.socket = socket

    def load(self, image1_name, image2_name):
        # load from load
        image1, image2 = self.loader.load(image1_name, image2_name)

        # overwrite with network if image_path is "network"
        if image1_name == "network":
            image1, image2_candidate = self.load_image()
            if image2_name == "network" and image2_candidate.shape[0] > 0:
                image2 = image2_candidate
        return image1, image2

    def load_image(self):
        print(f"Image Loader listending ...")
        msgs = self.socket.recv_multipart(0)
        if  len(msgs) == 2:
            # load image
            image_size = np.frombuffer(msgs[0], dtype=np.int32)
            width = image_size[0]
            height = image_size[1]
            print(f"width={width}, height={height}")
            msg = msgs[1]
            image = np.frombuffer(msg, dtype=np.uint8).reshape(height, width, -1).squeeze()
            return image, np.array([])
        elif len(msgs) == 4:
            # load image1
            image_size = np.frombuffer(msgs[0], dtype=np.int32)
            width = image_size[0]
            height = image_size[1]
            print(f"width={width}, height={height}")
            msg = msgs[1]
            image1 = np.frombuffer(msg, dtype=np.uint8).reshape(height, width, -1).squeeze()
            # load image2
            image_size = np.frombuffer(msgs[2], dtype=np.int32)
            width = image_size[0]
            height = image_size[1]
            msg = msgs[3]
            image2 = np.frombuffer(msg, dtype=np.uint8).reshape(height, width, -1).squeeze()
            return image1, image2
        else:
            print(f"Unexpected message: {msgs}")
            return np.array([]), np.array([])
