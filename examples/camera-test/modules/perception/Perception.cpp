#include "Perception.hpp"

/*Perception::Perception(){
	//FastScnn *segNetwork = new FastScnn(segModelPath);
	LogInfo("After serialization\n");
}*/

Perception::~Perception()
{
	// delete segNetwork;
}

int Perception::InitModule()
{
	/*
	 * initialize segmentation network
	 */
	/*int status_seg = seg_network.initEngine();
	if (!status_seg)
	{
		LogError("Perception: Failed to init fast scnn model\n");
		return 1;
	}*/

	/*
	 * create detection network
	 */
	int status_det = det_network.InitEngine();
	if (!status_det)
	{
		LogError("Perception: Failed to init yolo model\n");
		return 1;
	}
	return 0;
}
int Perception::RunPerception(pixelType *imgInput, pixelType *imgOutput)
{
	/*seg_network.process(imgInput, 1920, 1080); // 60ms 62ms(Paddle)

	int lane_center = seg_network.getLaneCenter(1); // 57ms

	seg_network.getRGB(*imgOutput, lane_center); // 54 ms
	*/
	// seg_network.PrintProfilerTimes();

	det_network.PreProcess(imgInput);
	det_network.Process();	   // run inference (22 ms)
	det_network.PostProcess(); // nms (very fast)

#if VISUALIZATION_ENABLED
	GetVisImage(imgInput);
	det_network.OverlayBBoxesOnVisImage(det_vis_image, DETECTION_ROI_W, DETECTION_ROI_H);
	OverlayVisImage(imgOutput);
#endif
}

void Perception::GetVisImage(pixelType *img_input)
{
	getROIOfImage(img_input, det_vis_image, CAMERA_INPUT_W, CAMERA_INPUT_H, DETECTION_ROI_W, DETECTION_ROI_H);
}

void Perception::OverlayVisImage(pixelType *img_output)
{
    for (int y = 0; y < DETECTION_ROI_H; ++y) {
        for (int x = 0; x < DETECTION_ROI_W; ++x) {
            // Calculate position in source and destination arrays
            int srcPos = y * DETECTION_ROI_W + x;
            int dstPos = (y + 512) * 1024 + x;

            // Copy pixel
            img_output[dstPos] = det_vis_image[srcPos];
        }
    }
}
