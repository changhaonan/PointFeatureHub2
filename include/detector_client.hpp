#pragma once
#include <zmq.h>
#include <cppzmq/zmq.hpp>
#include <cppzmq/zmq_addon.hpp>
// Declare below
class DetectorClient;

#ifdef DETECTOR_CLIENT_IMPLEMENTATION
// Implement below
class DetectorClient
{
public:
    DetectorClient(const int port);
    ~DetectorClient();
    void init();
    void run();
    void stop();

private:
    int port_;
    zmq::context_t context_;
    zmq::socket_t socket_;
    std::shared_ptr<ColorCamera> color_camera_ptr_;
};

#endif