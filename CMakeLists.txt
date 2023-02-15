cmake_minimum_required(VERSION 3.5)

# =============================================================================
project(PointFeatureHub LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/third_party")

add_subdirectory(test)