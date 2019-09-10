#!/bin/bash
#
# this script benchmarks DNN models from jetson-inference
# using the TensorRT trtexec tool and logs the results.
#
# usage: ./benchmark-models.sh <log-dir> <iterations> <runs>
#
# trtexec will profile the execution time of the network
# over N iterations, each iteration averaging over M runs.
#
# If unspecified, the default number of iterations is 10.
# If unspecified, the default number of average runs is 10.
#
# If the output log directory is left unspecified, the logs
# will be saved under the benchmark_logs/ directory.
#
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NETWORKS="$ROOT/../data/networks"
TRT_EXEC=/usr/src/tensorrt/bin/trtexec

LOG_DIR=$1
ITERATIONS=$2
AVG_RUNS=$3

if [ -z "$LOG_DIR" ]; then
	LOG_DIR="benchmark_logs"
fi

if [ -z "$ITERATIONS" ]; then
	ITERATIONS="10"
fi

if [ -z "$AVG_RUNS" ]; then
	AVG_RUNS="10"
fi

mkdir $LOG_DIR

function benchmark_onnx()
{
	model_dir=$1
	model_name=$2
	output_layer=$3
	
	if [ -z "$model_name" ]; then
		model_name="fcn_resnet18.onnx"
	fi

	if [ -z "$output_layer" ]; then
		output_layer="output_0"
	fi

	$TRT_EXEC --onnx=$NETWORKS/$model_dir/$model_name --output=$output_layer --iterations=$ITERATIONS --avgRuns=$AVG_RUNS --fp16 | tee $LOG_DIR/$model_dir.txt
}

benchmark_onnx "FCN-ResNet18-Cityscapes-512x256"
benchmark_onnx "FCN-ResNet18-Cityscapes-1024x512"
benchmark_onnx "FCN-ResNet18-Cityscapes-2048x1024"

benchmark_onnx "FCN-ResNet18-DeepScene-576x320"
benchmark_onnx "FCN-ResNet18-DeepScene-864x480"

benchmark_onnx "FCN-ResNet18-MHP-512x320"
benchmark_onnx "FCN-ResNet18-MHP-640x360"

benchmark_onnx "FCN-ResNet18-Pascal-VOC-320x320"
benchmark_onnx "FCN-ResNet18-Pascal-VOC-512x320"

benchmark_onnx "FCN-ResNet18-SUN-RGBD-512x400"
benchmark_onnx "FCN-ResNet18-SUN-RGBD-640x512"






