#ifndef _UTILS_
#define _UTILS_

#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fcntl.h>  // File control
#include <termios.h> // Terminal control
#include <unistd.h>  // UNIX standard
#include <cstring>   // For string functions
#include "../yololayer.h"

using namespace cv;

size_t sizeDims(const nvinfer1::Dims &dims, const size_t elementSize);
Mat floatToMat(float* inputImg, int width, int height);

int open_uart(const char* port, int baud_rate);
bool write_uart(int fd, const char* data);
int uart_reader(int fd);
bool getParkCommand(int fd);
bool getParkingDoneCommand(int fd);

bool sendMetaData(int fd, int laneCenter, std::vector<Yolo::Detection> res);

#endif