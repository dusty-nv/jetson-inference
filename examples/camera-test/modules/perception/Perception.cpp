#include "Perception.hpp"


/*Perception::Perception(){
    //FastScnn *segNetwork = new FastScnn(segModelPath);
	LogInfo("After serialization\n");
}*/

Perception::~Perception(){
    //delete segNetwork;
}

int Perception::initModule(){
    /*
	 * initialize segmentation network
	 */
	/*int status = segNetwork.initEngine();
	if (!status)
	{
		LogError("Perception: Failed to init fast scnn model\n");
		return 1;
	}
	*/
	
    /*
	 * create detection network
	 */
	int status = detNetwork.initEngine();
	if (!status)
	{
		LogError("Perception: Failed to init yolo model\n");
		return 1;
	}
	return 0;
}
int Perception::runPerception(pixelType *imgInput, pixelType **imgOutput){
    //segNetwork.process(imgInput, 1920, 1080); // 60ms 62ms(Paddle)

    //int lane_center = segNetwork.getLaneCenter(1); // 57ms

    //segNetwork.getRGB(*imgOutput, lane_center); // 54 ms

    //segNetwork.PrintProfilerTimes();

    // std::vector<Yolo::Detection> res;

    // det.process(imgInput, 1920, 1080); // 22 ms

	// det.getRgb(imgInput, imgOutput, 1920, 1080, 540, 540);
		
	// nms(res, (float *)det.mBindings[1]); // 3us
}