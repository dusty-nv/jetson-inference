#pragma once

#include "Fastscnn.hpp"
#include "YoloV3.hpp"
#include "ProjectPaths.h"

/* Configuration of the visualization window */
#define VISUALIZATION_ENABLED true
#define VIS_WINDOW_W SEG_MAP_W
#define VIS_WINDOW_H DETECTION_ROI_H+SEG_MAP_H


class Perception
{
    public:
    /*
    ** Perception module constructor
    */
    Perception():seg_network(segModelPath),det_network(detModelPath){};

    /*
    ** Perception module desstructor
    */
    ~Perception();

    /*
    ** Initialize the module
    */
    int InitModule();
    int RunPerception(pixelType *imgInput, pixelType **imgOutput);

    protected:
    FastScnn seg_network;
    YoloV3 det_network;
};