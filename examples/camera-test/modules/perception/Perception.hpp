#pragma once

#include "Fastscnn.hpp"
#include "YoloV3.hpp"
#include "ProjectPaths.h"

class Perception
{
    public:
    /*
    ** Perception module constructor
    */
    Perception():segNetwork(segModelPath),detNetwork(detModelPath){};

    /*
    ** Perception module desstructor
    */
    ~Perception();

    /*
    ** Initialize the module
    */
    int initModule();
    int runPerception(pixelType *imgInput, pixelType **imgOutput);

    protected:
    FastScnn segNetwork;
    YoloV3 detNetwork;
};