/*
 * http://github.com/dusty-nv/jetson-inference
 */

#ifndef __GIE_CAFFE_H
#define __GIE_CAFFE_H


#include "Infer.h"
#include "caffeParser.h"
#include "logGIE.h"


/**
 * Create an optimized GIE network from caffe prototxt and model file.
 */
bool caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 std::ostream& gieModelStream)				// output stream for the GIE model
{
	// create API root class - must span the lifetime of the engine usage
	nvinfer1::IBuilder* builder = createInferBuilder(gLogger);
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	// parse the caffe model to populate the network, then set the outputs
	nvcaffeparser1::CaffeParser* parser = new nvcaffeparser1::CaffeParser;

	const bool useFp16 = builder->plaformHasFastFp16();
	printf(LOG_GIE "platform %s FP16 support.\n", useFp16 ? "has" : "does not have");
	printf(LOG_GIE "loading %s %s\n", deployFile.c_str(), modelFile.c_str());
	
	nvinfer1::DataType modelDataType = useFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; // create a 16-bit model if it's natively supported
	const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor =
		parser->parse(deployFile.c_str(),		// caffe deploy file
					  modelFile.c_str(),		// caffe model file
					 *network,					// network definition that the parser will populate
					  modelDataType);

	if( !blobNameToTensor )
	{
		printf(LOG_GIE "failed to parse caffe network\n");
		return false;
	}
	
	// the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate	
	const size_t num_outputs = outputs.size();
	
	for( size_t n=0; n < num_outputs; n++ )
		network->markOutput(*blobNameToTensor->find(outputs[n].c_str()));


	// Build the engine
	printf(LOG_GIE "configuring CUDA engine\n");
		
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(16 << 20);

	// set up the network for paired-fp16 format, only on DriveCX
	if(useFp16)
		builder->setHalf2Mode(true);

	printf(LOG_GIE "building CUDA engine\n");
	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
	
	if( !engine )
	{
		printf(LOG_GIE "failed to build CUDA engine\n");
		return false;
	}

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	delete parser;

	// serialize the engine, then close everything down
	engine->serialize(gieModelStream);
	engine->destroy();
	builder->destroy();
	
	return true;
}


#endif
