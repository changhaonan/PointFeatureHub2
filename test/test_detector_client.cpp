#define DETECTOR_CLIENT_IMPLEMENTATION
#include "detector_client.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>

int main(int argc, char **argv)
{
    auto current_path = std::filesystem::current_path();
    auto image_path_list = {current_path / "data/image_train/0.png", current_path / "data/image_train/1.png"};
    pfh::DetectorClient detector_client(9090, 400);
    detector_client.SetUp();
    for (auto image_path : image_path_list)
    {
        cv::Mat image = cv::imread(image_path.string());
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detector_client.Detect(image, Eigen::Vector4f(100, 200, 100, 200), 0, keypoints, descriptors);
        // Draw keypoints
        cv::Mat image_with_keypoints;
        cv::drawKeypoints(image, keypoints, image_with_keypoints);
        cv::imshow("image_with_keypoints", image_with_keypoints);
        cv::waitKey(0);
    }

    return 0;
}