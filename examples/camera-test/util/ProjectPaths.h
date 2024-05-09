#ifndef _PROJECTPATHS_
#define _PROJECTPATHS_

#include <string>

const std::string projectPath = "/home/jeet/ideas/myjetsonrepo/jetson-inference/examples/camera-test/";
const std::string trtModelsPath = projectPath + "models/";

const std::string segModelPath = trtModelsPath + "fastscnn_unity.trt";
const std::string detModelPath = trtModelsPath + "yolov3-tiny.engine";

#endif