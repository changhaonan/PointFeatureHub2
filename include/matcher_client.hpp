#pragma once
#include <Eigen/Dense>
#include <zmq.h>
#include <cppzmq/zmq.hpp>
#include <cppzmq/zmq_addon.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <memory>
// Declare below

namespace pfh
{
    class MatcherClient
    {
    public:
        MatcherClient(const int port, const int fixed_size);
        ~MatcherClient();
        bool SetUp();
        void Match(const cv::Mat &image1, const cv::Mat &image2,
                   const Eigen::Vector4f &roi1, const Eigen::Vector4f &roi2,
                   const float rot_deg1, const float rot_deg2,
                   std::vector<cv::KeyPoint> &keypoints1,
                   std::vector<cv::KeyPoint> &keypoints2);
        // Utility functions
        void TransformImage(
            const cv::Mat &image, const Eigen::Vector4f &roi, const float rot_deg, cv::Mat &transformed_image,
            Eigen::Matrix3f &forward_transform);

    private:
        int fixed_size_;
        int port_;
        zmq::context_t context_;
        zmq::socket_t socket_;
    };

    // #ifdef MATCHER_CLIENT_IMPLEMENTATION
    // Implement below
    MatcherClient::MatcherClient(const int port, const int fixed_size)
        : port_(port), fixed_size_(fixed_size), context_(1),
          socket_(context_, ZMQ_REQ){};

    MatcherClient::~MatcherClient()
    {
        socket_.close();
        context_.close();
    };

    bool MatcherClient::SetUp()
    {
        socket_.connect("tcp://0.0.0.0:" + std::to_string(port_));
        std::cout << "Image connected to port " << port_ << ", Keypoint connected to port " << port_ << "..." << std::endl;
        return true;
    };

    void MatcherClient::Match(const cv::Mat &image1, const cv::Mat &image2,
                              const Eigen::Vector4f &roi1, const Eigen::Vector4f &roi2,
                              const float rot_deg1, const float rot_deg2,
                              std::vector<cv::KeyPoint> &keypoints1,
                              std::vector<cv::KeyPoint> &keypoints2)
    {
        cv::Mat transformed_image1, transformed_image2;
        Eigen::Matrix3f forward_transform1, forward_transform2;
        TransformImage(image1, roi1, rot_deg1, transformed_image1, forward_transform1);
        TransformImage(image2, roi2, rot_deg2, transformed_image2, forward_transform2);

        {
            zmq::message_t msg(2 * sizeof(int));
            std::vector<int> wh = {transformed_image1.cols, transformed_image1.rows};
            std::memcpy(msg.data(), wh.data(), 2 * sizeof(int));
            socket_.send(msg, zmq::send_flags::sndmore);
        }
        {
            cv::Mat flat = transformed_image1.reshape(1, transformed_image1.total() * transformed_image1.channels());
            std::vector<unsigned char> vec = transformed_image1.isContinuous() ? flat : flat.clone();
            zmq::message_t msg(vec.size() * sizeof(unsigned char));
            std::memcpy(msg.data(), vec.data(), vec.size() * sizeof(unsigned char));
            socket_.send(msg, zmq::send_flags::sndmore);
        }
        {
            zmq::message_t msg(2 * sizeof(int));
            std::vector<int> wh = {transformed_image2.cols, transformed_image2.rows};
            std::memcpy(msg.data(), wh.data(), 2 * sizeof(int));
            socket_.send(msg, zmq::send_flags::sndmore);
        }
        {
            cv::Mat flat = transformed_image2.reshape(1, transformed_image2.total() * transformed_image2.channels());
            std::vector<unsigned char> vec = transformed_image2.isContinuous() ? flat : flat.clone();
            zmq::message_t msg(vec.size() * sizeof(unsigned char));
            std::memcpy(msg.data(), vec.data(), vec.size() * sizeof(unsigned char));
            socket_.send(msg, zmq::send_flags::none);
        }

        printf("[detector client]: waiting for reply\n");
        std::vector<zmq::message_t> recv_msgs;
        auto result = zmq::recv_multipart(socket_, std::back_inserter(recv_msgs));
        printf("[detector client]: got reply\n");

        std::vector<int> info(2);
        std::memcpy(info.data(), recv_msgs[0].data(), info.size() * sizeof(int));
        const int num_match = info[0];
        const int feat_dim = info[1];

        if (num_match == 0)
        {
            std::cout << "[detector client]: no feature detected." << std::endl;
            return;
        }

        std::vector<float> kpts1_array(num_match * 2);
        std::memcpy(kpts1_array.data(), recv_msgs[1].data(), kpts1_array.size() * sizeof(float));

        std::vector<float> kpts2_array(num_match * 2);
        std::memcpy(kpts2_array.data(), recv_msgs[2].data(), kpts2_array.size() * sizeof(float));

        keypoints1.resize(num_match);
        keypoints2.resize(num_match);
        Eigen::Matrix3f backward_transform1 = forward_transform1.inverse();
        Eigen::Matrix3f backward_transform2 = forward_transform2.inverse();
        std::cout << "backward_transform1: " << std::endl;
        std::cout << backward_transform1 << std::endl;
        std::cout << "backward_transform2: " << std::endl;
        std::cout << backward_transform2 << std::endl;

        for (auto i = 0; i < num_match; i++)
        {
            Eigen::Vector3f p1(kpts1_array[2 * i], kpts1_array[2 * i + 1], 1);
            p1 = backward_transform1 * p1;
            cv::KeyPoint kpt1({p1(0), p1(1)}, 1);
            keypoints1[i] = kpt1;

            Eigen::Vector3f p2(kpts2_array[2 * i], kpts2_array[2 * i + 1], 1);
            p1 = backward_transform2 * p2;
            cv::KeyPoint kpt2({p2(0), p2(1)}, 1);
            keypoints2[i] = kpt2;
        }
    };

    void MatcherClient::TransformImage(
        const cv::Mat &image, const Eigen::Vector4f &roi, const float rot_deg, cv::Mat &transformed_image,
        Eigen::Matrix3f &forward_transform)
    {
        // Check if image is empty
        if (image.empty())
        {
            std::cout << "Image is empty" << std::endl;
            throw std::runtime_error("Image is empty!");
            return;
        }
        // Check if roi is valid or roi2 is valid
        if (roi(0) < 0 || roi(1) > image.cols || roi(2) < 0 || roi(3) > image.rows)
        {
            std::cout << "ROI is invalid" << std::endl;
            throw std::runtime_error("ROI is invalid!");
            return;
        }

        const int W = roi(1) - roi(0);
        const int H = roi(3) - roi(2);
        forward_transform = Eigen::Matrix3f::Identity();
        Eigen::Matrix3f new_transform(Eigen::Matrix3f::Identity());
        int side = std::max(H, W);
        cv::Mat img = cv::Mat::zeros(side, side, CV_8UC3);
        for (int h = 0; h < H; h++)
        {
            for (int w = 0; w < W; w++)
            {
                img.at<cv::Vec3b>(h, w) = image.at<cv::Vec3b>(h + roi(2), w + roi(0));
            }
        }
        new_transform.setIdentity();
        new_transform(0, 2) = -roi(0);
        new_transform(1, 2) = -roi(2);
        forward_transform = new_transform * forward_transform;
        if (rot_deg != 0)
        {
            cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(side / 2, side / 2), rot_deg, 1);
            cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), img.size(), rot_deg).boundingRect2f();
            M.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
            M.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;
            side = std::max(bbox.width, bbox.height);
            cv::warpAffine(img, img, M, {side, side});
            Eigen::Matrix<float, 2, 3> tmp;
            cv::cv2eigen(M, tmp);
            new_transform.setIdentity();
            new_transform.block(0, 0, 2, 3) = tmp;
            forward_transform = new_transform * forward_transform;
        }
        const int H_input = fixed_size_;
        const int W_input = fixed_size_;
        cv::resize(img, transformed_image, {W_input, H_input});
        new_transform.setIdentity();
        new_transform(0, 0) = W_input / float(side);
        new_transform(1, 1) = H_input / float(side);
        forward_transform = new_transform * forward_transform;
    }
    // #endif

}
