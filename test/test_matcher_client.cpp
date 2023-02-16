#define MATCHER_CLIENT_IMPLEMENTATION
#include "matcher_client.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>

int main(int argc, char **argv)
{
    auto current_path = std::filesystem::current_path();
    std::vector<std::filesystem::path> image_path_list = {current_path / "data/image_train/0.png", current_path / "data/image_train/1.png"};
    pfh::MatcherClient matcher_client(5050, 400);
    matcher_client.SetUp();

    cv::Mat image1 = cv::imread(image_path_list[0].string());
    cv::Mat image2 = cv::imread(image_path_list[0].string());
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    matcher_client.Match(
        image1, image2,
        Eigen::Vector4f(0, image1.cols, 0, image1.rows),
        Eigen::Vector4f(0, image2.cols, 0, image2.rows),
        0, 90,
        keypoints1, keypoints2);

    // Draw matches
    cv::Mat image_with_matches;
    // Use a full DMat to draw all matches
    std::vector<cv::DMatch> matches;
    for (auto i = 0; i < keypoints1.size(); i++)
    {
        cv::DMatch match(i, i, 0);
        matches.push_back(match);
    }
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, image_with_matches);
    cv::imshow("image_with_matches", image_with_matches);
    cv::waitKey(0);
    return 0;
}