set(LIBRARIES Eigen3::Eigen ${OpenCV_LIBRARIES} zmq)

add_executable(test_detector_client test_detector_client.cpp)
target_link_libraries(test_detector_client ${LIBRARIES})

add_executable(test_matcher_client test_matcher_client.cpp)
target_link_libraries(test_matcher_client ${LIBRARIES})