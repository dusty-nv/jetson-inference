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
int Perception::RunPerception(pixelType *imgInput, pixelType **imgOutput)
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
	det_network.OverlayROI(imgInput, *imgOutput, 1920, 1080, VIS_WINDOW_W, VIS_WINDOW_H);
	det_network.OverlayBBoxesOnROI(*imgOutput,0,0,0,0);
#endif
}