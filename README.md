# PointFeatureHub

This works intends to compare the result from the different pointfeature source & compare their performance.

Currently included:

## Decteting method:

- SuperPoint
- R2D2
- OpenCV-based
    - ORB
    - SIFT

## Matching method:

- SuperGLUE
- OpenCV: BF-matcher
- AdaLAM from kernia
- Detector-Free
    - LoFTR from kernia
- MAGSAC
- OnePose (3D to 2D matching)

## Architecture

The current architecture we used is based ZMQ socket. The reason why we should this is because this can help us to integrate different conda environment, python and c++. Make the whole structure easy to use. 

We also provide python and C++ wrapper to integrate with your own projects.

## TODO:

- [ ] Add python client.
- [ ] Add CUDA-based data passing support.
- [ ] Add OnePose to this structure.
    - [ ] Add OnePose method
    - [ ] Add network client & wrapper