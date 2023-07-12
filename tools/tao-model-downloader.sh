#!/bin/bash
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

LOG="[TRT]   "
WGET_QUIET="--quiet"
NETWORK_DIR="/usr/local/bin/networks"

source /usr/local/bin/l4t_version.sh


#
# try to download a file from URL
#
function attempt_download_file()
{
	local filename=$1
	local URL=$2
	
	wget $WGET_QUIET --show-progress --progress=bar:force:noscroll --no-check-certificate $URL -O $filename
	
	local wget_status=$?

	if [ $wget_status != 0 ]; then
		echo "$LOG wget failed to download '$filename' (error code=$wget_status)"
		return $wget_status
	fi

	return 0
}


#
# download a file from URL
#
function download_file()
{
	local filename=$1
	local URL=$2
	local retries=0
	
	WGET_QUIET="--quiet"

	while [ $retries -lt 10 ]; do
		attempt_download_file $filename $URL

		local download_status=$?

		if [ $download_status = 0 ]; then
			return 0
		fi

		((retries++))
		echo "$LOG attempting to retry download of $URL  (retry $retries of 10)"
		WGET_QUIET="--verbose"
	done
	
	echo "$LOG failed to download $URL"
	exit 1
}


#
# try to download/extract an archive
#
function attempt_download_archive()
{
	local filename=$1
	local URL=$2
	
	wget $WGET_QUIET --show-progress --progress=bar:force:noscroll --no-check-certificate $URL -O $filename
	
	local wget_status=$?

	if [ $wget_status != 0 ]; then
		echo "$LOG wget failed to download '$filename' (error code=$wget_status)"
		return $wget_status
	fi

	tar -xzf $filename -C $OUTPUT_DIR

	local tar_status=$?

	if [ $tar_status != 0 ]; then
		echo "$LOG tar failed to extract '$filename' (error code=$tar_status)"
		return $tar_status
	fi

	rm $filename
	return 0
}


#
# download/extract an archive
#
function download_archive()
{
	local filename=$1
	local URL=$2
	local retries=0
	
	WGET_QUIET="--quiet"

	while [ $retries -lt 10 ]; do
		attempt_download_archive $filename $URL

		local download_status=$?

		if [ $download_status = 0 ]; then
			return 0
		fi

		((retries++))
		echo "$LOG attempting to retry download of $URL  (retry $retries of 10)"
		WGET_QUIET="--verbose"
	done
	
	echo "$LOG failed to download $URL"
	exit 1
}


#
# download the correct tao-converter tool for the platform
#
function download_tao_converter()
{
	local url="https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v3.21.11_trt8.0_aarch64/files/tao-converter"
	
	if [[ $L4T_RELEASE -ge 34 ]]; then
		url="https://api.ngc.nvidia.com/v2/resources/nvidia/tao/tao-converter/versions/v3.22.05_trt8.4_aarch64/files/tao-converter"
	fi
	
	echo "$LOG downloading tao-converter from $url"
	download_file "tao-converter" $url
	chmod +x tao-converter
}


#
# convert a TAO .etlt model to TensorRT engine
#
function tao_to_trt()
{
	local model_input="$1"
	local model_output="$model_input.engine"
	local calibration="$2"
	local encryption_key=${3:-"tlt_encode"}
	local input_dims=${4:-"3,544,960"}
	local output_layers=${5:-"output_bbox/BiasAdd,output_cov/Sigmoid"}
	
	local max_batch_size="1"
	local workspace_size="4294967296" # 4GB 
	local precision="int8"
	
	if [[ $L4T_RELEASE -lt 34 ]]; then
		precision="fp16"  # use FP16 on JetPack4 in lieu of detecting GPU compute capabilities
	fi
	
	download_tao_converter
	
	echo "detectNet -- converting TAO model to TensorRT engine:"
	echo "          -- input          $model_input"
	echo "          -- output         $model_output"
	echo "          -- calibration    $calibration"
	echo "          -- encryption_key $encryption_key"
	echo "          -- input_dims     $input_dims"
	echo "          -- output_layers  $output_layers"
	echo "          -- max_batch_size $max_batch_size"
	echo "          -- workspace_size $workspace_size"
	echo "          -- precision      $precision"
	
	./tao-converter \
		-k $encryption_key \
		-d $input_dims \
		-o $output_layers \
		-m $max_batch_size \
		-w $workspace_size \
		-t $precision \
		-c $calibration \
		-e $model_output \
		$model_input
		
	local convert_status=$?
	
	if [ $convert_status != 0 ]; then
		echo "$LOG failed to convert model '$model_input' to TensorRT..."
		exit 1
	fi
	
	if [ ! -f $model_output ]; then
		echo "$LOG missing output model '$model_output'"
		exit 1
	fi
	
	echo "$LOG successfully built TensorRT engine '$model_output'"
}


#
# PeopleNet
#
function download_peoplenet_v261()
{
	local model_name="peoplenet_deployable_quantized_v2.6.1"
	local model_path="$NETWORK_DIR/$model_name"

	mkdir $model_path
	cd $model_path

	download_file "resnet34_peoplenet_int8.etlt" "https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_v2.6.1/files/resnet34_peoplenet_int8.etlt"
	download_file "resnet34_peoplenet_int8.txt" "https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_v2.6.1/files/resnet34_peoplenet_int8.txt"
	download_file "labels.txt" "https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_v2.6.1/files/labels.txt"
	download_file "colors.txt" "https://nvidia.box.com/shared/static/s5ok5wgf2rn38jhj7zi0x9e8fw0wqnyr.txt"
	
	tao_to_trt "resnet34_peoplenet_int8.etlt" "resnet34_peoplenet_int8.txt"
}

function download_peoplenet_v232()
{
	local model_name="peoplenet_pruned_quantized_v2.3.2"
	local model_path="$NETWORK_DIR/$model_name"

	mkdir $model_path
	cd $model_path

	download_file "resnet34_peoplenet_pruned_int8.etlt" "https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_v2.3.2/files/resnet34_peoplenet_pruned_int8.etlt"
	download_file "resnet34_peoplenet_pruned_int8.txt" "https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_v2.3.2/files/resnet34_peoplenet_pruned_int8.txt"
	download_file "labels.txt" "https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_v2.3.2/files/labels.txt"
	download_file "colors.txt" "https://nvidia.box.com/shared/static/s5ok5wgf2rn38jhj7zi0x9e8fw0wqnyr.txt"
	
	tao_to_trt "resnet34_peoplenet_pruned_int8.etlt" "resnet34_peoplenet_pruned_int8.txt"
}


#
# DashCamNet
#
function download_dashcamnet_v103()
{
	local model_name="dashcamnet_pruned_v1.0.3"
	local model_path="$NETWORK_DIR/$model_name"

	mkdir $model_path
	cd $model_path

	download_file "resnet18_dashcamnet_pruned.etlt" "https://api.ngc.nvidia.com/v2/models/nvidia/tao/dashcamnet/versions/pruned_v1.0.3/files/resnet18_dashcamnet_pruned.etlt"
	download_file "dashcamnet_int8.txt" "https://api.ngc.nvidia.com/v2/models/nvidia/tao/dashcamnet/versions/pruned_v1.0.3/files/dashcamnet_int8.txt"
	download_file "labels.txt" "https://api.ngc.nvidia.com/v2/models/nvidia/tao/dashcamnet/versions/pruned_v1.0.3/files/labels.txt"
	download_file "colors.txt" "https://nvidia.box.com/shared/static/lv953cgp4klkbqtcbdwb67cexc8uqlam.txt"
	
	tao_to_trt "resnet18_dashcamnet_pruned.etlt" "dashcamnet_int8.txt"
}


#
# TrafficCamNet
#
function download_trafficcamnet_v103()
{
	local model_name="trafficcamnet_pruned_v1.0.3"
	local model_path="$NETWORK_DIR/$model_name"

	mkdir $model_path
	cd $model_path

	download_file "resnet18_trafficcamnet_pruned.etlt" "https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet/versions/pruned_v1.0.3/files/resnet18_trafficcamnet_pruned.etlt"
	download_file "trafficcamnet_int8.txt" "https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet/versions/pruned_v1.0.3/files/trafficcamnet_int8.txt"
	download_file "labels.txt" "https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet/versions/pruned_v1.0.3/files/labels.txt"
	download_file "colors.txt" "https://nvidia.box.com/shared/static/8ed18jkxrm8ya639pnh0eem91ue52roc.txt"
	
	tao_to_trt "resnet18_trafficcamnet_pruned.etlt" "trafficcamnet_int8.txt"
}


#
# FaceDetect
#
function download_facedetect_v201()
{
	local model_name="facedetect_pruned_quantized_v2.0.1"
	local model_path="$NETWORK_DIR/$model_name"

	mkdir $model_path
	cd $model_path

	download_file "resnet18_facedetect_pruned.etlt" "https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0.1/files/model.etlt"
	download_file "int8_calibration.txt" "https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0.1/files/int8_calibration.txt"
	download_file "labels.txt" "https://nvidia.box.com/shared/static/i7bkwk44ttiut6ak8qqb3fxcqsxz6vvg.txt"
	download_file "colors.txt" "https://nvidia.box.com/shared/static/u0sp4kede1ypd0ekqj9sd12pbtzxts0g.txt"
	
	tao_to_trt "resnet18_facedetect_pruned.etlt" "int8_calibration.txt" "nvidia_tlt" "3,416,736"
}



#
# command-line parsing
#
MODEL="$1"

echo "$LOG downloading $MODEL"

if [[ "$MODEL" == "peoplenet_deployable_quantized_v2.6.1" ]]; then
	download_peoplenet_v261
elif [[ "$MODEL" == "peoplenet_pruned_quantized_v2.3.2" ]]; then
	download_peoplenet_v232
elif [[ "$MODEL" == "dashcamnet_pruned_v1.0.3" ]]; then
	download_dashcamnet_v103
elif [[ "$MODEL" == "trafficcamnet_pruned_v1.0.3" ]]; then
	download_trafficcamnet_v103
elif [[ "$MODEL" == "facedetect_pruned_quantized_v2.0.1" ]]; then
	download_facedetect_v201
else
	echo "$LOG error -- invalid or unknown model selected ($MODEL)"
	exit 1
fi

exit 0
