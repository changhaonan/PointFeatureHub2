#define DETECTOR_CLIENT_IMPLEMENTATION
#include "detector_client.hpp"


int main(int argc, char **argv)
{
    DetectorClient detector_client(5555, 640, 480);
    detector_client.SetUp();
    return 0;
}