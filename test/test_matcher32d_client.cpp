#define MATCHER32D_CLIENT_IMPLEMENTATION
#include "matcher32d_client.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>

int main(int argc, char **argv)
{
    auto current_path = std::filesystem::current_path();
    std::vector<std::filesystem::path> image_path_list = {
        current_path / "data/2d_query/001/6.png",
        current_path / "data/2d_query/001/379.png"};
    pfh::Matcher32DClient matcher_client(9090, 400);
    matcher_client.SetUp();

    std::string sparse_model_path = (current_path / "data/3d_train/001").string();
    cv::Mat image = cv::imread(image_path_list[0].string());
    std::vector<cv::KeyPoint> kpts2d;
    std::vector<Eigen::Vector3f> kpts3d;
    kpts2d.clear();
    kpts3d.clear();
    Eigen::Matrix3f K;
    K << 5.986332104667836802e+02, 0.000000000000000000e+00, 3.245491486036403330e+02,
        0.000000000000000000e+00, 5.901477555297449271e+02, 2.237197338492138670e+02,
        0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00;
    matcher_client.Match32D(sparse_model_path, image, Eigen::Vector4f(0, image.cols * 0.5, 0, image.rows * 0.5), 0, K, kpts2d, kpts3d);

    // Draw matches
    cv::Mat image_with_matches;
    // Use a full DMat to draw all matches
    std::vector<cv::DMatch> matches;
    for (auto i = 0; i < kpts2d.size(); i++)
    {
        cv::DMatch match(i, i, 0);
        matches.push_back(match);
    }
    return 0;
}