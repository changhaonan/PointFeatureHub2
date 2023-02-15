import zmq
import numpy as np
import cv2


def use_socket(extract_func):
    def function_wrapper(image, socket):
        # detect image
        xys, desc, scores = extract_func(image)

        # send the results with socket
        num_feat = len(xys)
        feat_dim = desc.shape[1]
        print(f"num_feat={num_feat}, feat_dim={feat_dim}.")
        if num_feat == 0:
            msg = np.array([0, 0]).reshape(-1).astype(np.int32).tobytes()
            socket.send(msg, 0)
            print("No feature detected. Skip this frame.")
            return
        msg = np.array([num_feat, feat_dim]).reshape(-1).astype(np.int32).tobytes()
        socket.send(msg, 2)
        msg = xys.astype(np.float32).reshape(-1).tobytes()
        socket.send(msg, 2)
        msg = desc.astype(np.float32).reshape(-1).tobytes()
        socket.send(msg, 0)
        return xys, desc, scores
    return function_wrapper
