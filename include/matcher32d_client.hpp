#pragma once
#include <Eigen/Dense>
#include <zmq.h>
#include <cppzmq/zmq.hpp>
#include <cppzmq/zmq_addon.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <memory>
#include <string>
// Declare below

namespace pfh
{
    class Matcher32DClient
    {
    public:
        Matcher32DClient(const int port, const int fixed_size);
        ~Matcher32DClient();
        bool SetUp();
        void Match32D(const std::string &sparse_model_path,
                      const cv::Mat &image,
                      const Eigen::Vector4f &roi,
                      const float rot_deg,
                      const Eigen::Matrix3f &K,
                      std::vector<cv::KeyPoint> &kpts2d,
                      std::vector<Eigen::Vector3f> &kpts3d);
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

    // #ifdef MATCHER32D_CLIENT_IMPLEMENTATION
    // Implement below
    Matcher32DClient::Matcher32DClient(const int port, const int fixed_size)
        : port_(port), fixed_size_(fixed_size), context_(1),
          socket_(context_, ZMQ_REQ){};

    Matcher32DClient::~Matcher32DClient()
    {
        socket_.close();
        context_.close();
    };

    bool Matcher32DClient::SetUp()
    {
        socket_.connect("tcp://0.0.0.0:" + std::to_string(port_));
        std::cout << "Image connected to port " << port_ << ", Keypoint connected to port " << port_ << "..." << std::endl;
        return true;
    };

    void Matcher32DClient::Match32D(const std::string &sparse_model_path,
                                    const cv::Mat &image,
                                    const Eigen::Vector4f &roi,
                                    const float rot_deg,
                                    const Eigen::Matrix3f &K,
                                    std::vector<cv::KeyPoint> &kpts2d,
                                    std::vector<Eigen::Vector3f> &kpts3d)
    {
        // Get the first 3 channels of image
        cv::Mat image_3ch;
        if (image.channels() == 4)
            cv::cvtColor(image, image_3ch, cv::COLOR_RGBA2RGB);
        else
            image_3ch = image;

        int string_len = sparse_model_path.size();
        cv::Mat transformed_image;
        Eigen::Matrix3f forward_transform;
        TransformImage(image_3ch, roi, rot_deg, transformed_image, forward_transform);
        {
            zmq::message_t msg(2 * sizeof(int));
            std::vector<int> wh = {transformed_image.cols, transformed_image.rows};
            std::memcpy(msg.data(), wh.data(), 2 * sizeof(int));
            socket_.send(msg, zmq::send_flags::sndmore);
        }
        {
            // Send sparse model path string
            socket_.send(zmq::buffer(sparse_model_path), zmq::send_flags::sndmore);
        }
        {
            cv::Mat flat = transformed_image.reshape(1, transformed_image.total() * transformed_image.channels());
            std::vector<unsigned char> vec = transformed_image.isContinuous() ? flat : flat.clone();
            zmq::message_t msg(vec.size() * sizeof(unsigned char));
            std::memcpy(msg.data(), vec.data(), vec.size() * sizeof(unsigned char));
            socket_.send(msg, zmq::send_flags::sndmore);
        }
        {
            zmq::message_t msg(9 * sizeof(float));
            std::memcpy(msg.data(), K.data(), 9 * sizeof(float));
            socket_.send(msg, zmq::send_flags::none);
        }

        printf("[matcher32d client]: waiting for reply\n");
        std::vector<zmq::message_t> recv_msgs;
        auto result = zmq::recv_multipart(socket_, std::back_inserter(recv_msgs));
        printf("[matcher32d client]: got reply\n");

        std::vector<int> info(2);
        std::memcpy(info.data(), recv_msgs[0].data(), info.size() * sizeof(int));
        const int num_match = info[0];
        const int feat_dim = info[1];

        if (num_match == 0)
        {
            std::cout << "[matcher client]: no feature detected." << std::endl;
            return;
        }

        std::vector<float> kpts3d_array(num_match * 3);
        std::memcpy(kpts3d_array.data(), recv_msgs[1].data(), kpts3d_array.size() * sizeof(float));
        std::vector<float> kpts2d_array(num_match * 2);
        std::memcpy(kpts2d_array.data(), recv_msgs[2].data(), kpts2d_array.size() * sizeof(float));

        kpts2d.resize(num_match);
        Eigen::Matrix3f backward_transform = forward_transform.inverse();

        for (auto i = 0; i < num_match; i++)
        {
            Eigen::Vector3f p2d(kpts2d_array[2 * i], kpts2d_array[2 * i + 1], 1);
            p2d = backward_transform * p2d;
            cv::KeyPoint kpt({p2d(0), p2d(1)}, 1);
            kpts2d[i] = kpt;

            Eigen::Vector3f p3d(kpts3d_array[3 * i], kpts3d_array[3 * i + 1], kpts3d_array[3 * i + 2]);
            kpts3d.push_back(p3d);
        }
    };

    void Matcher32DClient::TransformImage(
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
        // Assert the channel
        assert(image.channels() == 3);

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
