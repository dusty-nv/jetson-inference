/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <signal.h>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#include "jetson-utils/videoSource.h"
#include "jetson-utils/videoOutput.h"

#include "util/utils.h"

#include "yolo/yolov3.h"
#include "fastscnn/fastscnn.h"

#define IMG_OUT_W 1024
#define IMG_OUT_H 512
#define UNITY_ENVIRONMENT 3

using namespace std;

bool signal_recieved = false;
bool toggleParking = false;

float iou(float lbox[4], float rbox[4])
{
	float interBox[] = {
		std::max(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), // left
		std::min(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), // right
		std::max(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), // top
		std::min(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), // bottom
	};

	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
		return 0.0f;

	float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Yolo::Detection &a, const Yolo::Detection &b)
{
	return a.det_confidence > b.det_confidence;
}

void nms(std::vector<Yolo::Detection> &res, float *output, float nms_thresh = NMS_THRESH)
{
	std::map<float, std::vector<Yolo::Detection>> m;
	for (int i = 0; i < output[0] && i < 1000; i++)
	{
		if (output[1 + 7 * i + 4] <= BBOX_CONF_THRESH)
			continue;
		Yolo::Detection det;
		memcpy(&det, &output[1 + 7 * i], 7 * sizeof(float));
		if (m.count(det.class_id) == 0)
			m.emplace(det.class_id, std::vector<Yolo::Detection>());
		m[det.class_id].push_back(det);
	}
	for (auto it = m.begin(); it != m.end(); it++)
	{
		// std::cout << it->second[0].class_id << " --- " << std::endl;
		auto &dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp);
		for (size_t m = 0; m < dets.size(); ++m)
		{
			auto &item = dets[m];
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n)
			{
				if (iou(item.bbox, dets[n].bbox) > nms_thresh)
				{
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}

cv::Rect get_rect(cv::Mat &img, float bbox[4])
{
	int l, r, t, b;
	float r_w = 320 / (img.cols * 1.0);
	float r_h = 320 / (img.rows * 1.0);
	if (r_h > r_w)
	{
		l = bbox[0] - bbox[2] / 2.f;
		r = bbox[0] + bbox[2] / 2.f;
		t = bbox[1] - bbox[3] / 2.f - (320 - r_w * img.rows) / 2;
		b = bbox[1] + bbox[3] / 2.f - (320 - r_w * img.rows) / 2;
		l = l / r_w;
		r = r / r_w;
		t = t / r_w;
		b = b / r_w;
	}
	else
	{
		l = bbox[0] - bbox[2] / 2.f - (320 - r_h * img.cols) / 2;
		r = bbox[0] + bbox[2] / 2.f - (320 - r_h * img.cols) / 2;
		t = bbox[1] - bbox[3] / 2.f;
		b = bbox[1] + bbox[3] / 2.f;
		l = l / r_h;
		r = r / r_h;
		t = t / r_h;
		b = b / r_h;
	}
	return cv::Rect(l, t, r - l, b - t);
}
bool saveArray(uint8_t *arr, const char *filename)
{
	std::ofstream outfile(filename, std::ios::binary | std::ios::out);

	if (!outfile.is_open())
	{
		std::cout << "Could not open file for writing." << std::endl;
		return false;
	}

	outfile.write(reinterpret_cast<char *>(arr), 1 * 512 * 1024 * sizeof(uint8_t));
	outfile.close();
	return true;
}

bool readArray(float *arr, const char *filename)
{
	ifstream infile(filename, ios::binary);
	if (!infile || arr == nullptr)
	{
		cout << "Cant open file\n";
		return false;
	}
	float temp;
	for (int j = 0; j < 3 * 512 * 1024; j++)
	{
		infile.read((char *)&temp, sizeof(float));
		arr[j] = temp;
	}
	infile.close();
	return false;
}

bool matToUchar3(Mat img, uchar3 *out, int width, int height)
{
	if (out == NULL)
	{
		return false;
	}
	for (int i = 0; i < height; i++)
	{
		Vec3b *row_ptr = img.ptr<Vec3b>(i);
		for (int j = 0; j < width; j++)
		{
			out[i * width + j].x = (unsigned char)row_ptr[j][0];
			out[i * width + j].y = (unsigned char)row_ptr[j][1];
			out[i * width + j].z = (unsigned char)row_ptr[j][2];
		}
	}
}
bool ucharToMat(uchar3 *img, Mat out, int width, int height)
{
	if (img == NULL)
	{
		return false;
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			out.at<Vec3b>(i, j)[0] = img[i * width + j].x;
			out.at<Vec3b>(i, j)[1] = img[i * width + j].y;
			out.at<Vec3b>(i, j)[2] = img[i * width + j].z;
		}
	}
}
void saveClassMap(uint8_t *classMap, int rows, int cols, const char *filename)
{
	std::ofstream outfile(filename, std::ios::binary);

	if (!outfile)
	{
		std::cerr << "Cannot open file.\n";
		return;
	}

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			outfile.write((char *)&classMap[i * cols + j], sizeof(uint8_t));
		}
	}

	outfile.close();
}

void sig_handler(int signo)
{
	if (signo == SIGINT)
	{
		LogVerbose("received SIGINT\n");
		signal_recieved = true;
	}
}

//
// segmentation buffers
//

pixelType *imgOutDet = NULL;
pixelType *imgOutput = NULL; // reference to one of the above three

int2 outputSize;
int status;
int main(int argc, char **argv)
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);

#if UNITY_ENVIRONMENT == 1
	int clientSocket = socket(AF_INET, SOCK_STREAM, 0);

	sockaddr_in serverAddress;
	serverAddress.sin_family = AF_INET;
	serverAddress.sin_port = htons(25001);
	serverAddress.sin_addr.s_addr = inet_addr("192.168.100.122");

	connect(clientSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress));
#elif UNITY_ENVIRONMENT == 2
	/*
	 * Open the uart
	 */
	int fd = open_uart("/dev/ttyTHS1", B115200);
	if (fd == -1)
	{
		LogError("Ideas: Failed to open uart. Exiting...\n");
		return 1; // Exit if opening failed
	}
#endif

	/*
	 * alloc space for seg visualization image
	 */
	if (!cudaAllocMapped(&imgOutput, make_int2(IMG_OUT_W, IMG_OUT_H)))
	{
		LogError("Ideas: Failed to allocate CUDA memory for out image\n");
		return 1;
	}

	/*
	 * attach signal handler
	 */
	if (signal(SIGINT, sig_handler) == SIG_ERR)
		LogError("can't catch SIGINT\n");

	/*
	 * create input stream
	 */
	videoSource *input = videoSource::Create(cmdLine, ARG_POSITION(0));

	if (!input)
	{
		LogError("Ideas: Failed to create input stream\n");
		return 1;
	}

	/*
	 * create output stream
	 */
	videoOutput *output = videoOutput::Create(cmdLine, ARG_POSITION(1));

	if (!output)
	{
		LogError("Ideas: Failed to create output stream\n");
		return 1;
	}

	/*
	 * create detection network
	 */
	/*YoloV3 det("models/yolov3-tiny.engine");
	status = det.initEngine();
	if (!status)
	{
		LogError("Ideas: Failed to init yolo model\n");
		return 1;
	}*/

	/*
	 * create segmentation network
	 */
	FastScnn net("models/fastscnn_unity.trt");
	status = net.initEngine();
	if (!status)
	{
		LogError("Ideas: Failed to init fast scnn model\n");
		return 1;
	}
	status = net.loadGrid();
	if (!status)
	{
		LogError("Ideas: Failed to load grid\n");
		return 1;
	}

	// Init performance metrics

	auto start = std::chrono::high_resolution_clock::now();
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	int lane_center = -1;
	while (!signal_recieved)
	{
		
		std::vector<Yolo::Detection> res;
		pixelType *imgInput = NULL;
		int status = 0;
		if (!input->Capture(&imgInput, &status))
		{
			if (status == videoSource::TIMEOUT)
				continue;

			break; // EOS
		}

		// det.process(imgInput, 1920, 1080); // 22 ms

		// det.getRgb(imgInput, imgOutput, 1920, 1080, 540, 540);
		start = std::chrono::high_resolution_clock::now();
		net.process(imgInput, 1920, 1080);				// 22 ms
		stop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
		cout<<"Time taken: "<<duration.count()<<endl;
		generateClassMap((float *)net.mOutputs[0].CUDA, net.mClassMap); // 50ms
		CUDA(cudaDeviceSynchronize());

		// nms(res, (float *)det.mBindings[1]); // 3us

		// if(toggleParking==false){//30 ms
		lane_center = net.getLaneCenter(1);
		// cout<<"Lane center: "<<lane_center<<endl;
		//}else if(toggleParking==true)
		//{
		// lane_center = net.getParkingDirection(0);
		//}

		net.getRGB(imgOutput, lane_center); // 47 ms

		if (output != NULL)
		{
			output->Render(imgOutput, 1024, 512);
			//  update status bar
			char str[256];
			// sprintf(str, "Latency(ms): %li, FPS: %f", duration.count(), 1000.0f / (duration.count()));
			// output->SetStatus(str);

			// check if the user quit
			if (!output->IsStreaming())
				break;
		}
		// lane_center = 560;
		#if UNITY_ENVIRONMENT==1
			const char* message = std::__cxx11::to_string(lane_center).c_str();
			send(clientSocket, message, strlen(message), 0);
		#elif UNITY_ENVIRONMENT==2
		bool uartStatus = sendMetaData(fd, lane_center, res);

		if (uartStatus == false)
		{
			LogError("Ideas: uart failed to send\n");
		}
		#endif
		// if(getParkCommand(fd)==true){
		//	toggleParking = true;
		// }
		// if(getParkingDoneCommand(fd)==true){
		//	toggleParking = false;
		// }
		
	}
	/*
	 * destroy resources
	 */
	LogVerbose("Ideas: Shutting down...\n");
#if UNITY_ENVIRONMENT == 1
	close(clientSocket);
#elif UNITY_ENVIRONMENT == 2
	close(fd);
#endif
	// reader_thread.join();
	SAFE_DELETE(input);
	SAFE_DELETE(output);

	CUDA_FREE_HOST(imgOutput);
	destroyAllWindows();
	LogVerbose("Ideas: Shutdown complete.\n");
	return 0;
}
