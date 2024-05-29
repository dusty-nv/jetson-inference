#pragma once

#include "Fastscnn.hpp"
#include "YoloV3.hpp"
#include "ProjectPaths.h"

/* Configuration of the visualization window */
#define VISUALIZATION_ENABLED true
#define VIS_WINDOW_W SEG_MAP_W
#define VIS_WINDOW_H DETECTION_ROI_H+SEG_MAP_H

/* Camera input dimensions */
#define CAMERA_INPUT_W 1920
#define CAMERA_INPUT_H 1080


class Perception
{
    public:
    /*
    ** Perception module constructor
    */
    Perception():seg_network(segModelPath),det_network(detModelPath)
    {
        if (!cudaAllocMapped(&det_vis_image, make_int2(DETECTION_ROI_W, DETECTION_ROI_H)))
        {
            LogError("Perception: Failed to allocate CUDA memory for detection vis image\n");
        }
    };

    /*
    ** Perception module desstructor
    */
    ~Perception();

    /*
    ** Initialize the module
    */
    int InitModule();
    int RunPerception(pixelType *img_input, pixelType *img_output);

    protected:
    FastScnn seg_network;
    YoloV3 det_network;

    pixelType *det_vis_image;

    /* Get the image thats fed to the detection network before preprocessing */
    void GetVisImage(pixelType *img_input);
    void OverlayVisImage(pixelType *img_output);
};