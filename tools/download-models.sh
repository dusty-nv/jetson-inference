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


APP_TITLE="Hello AI World (jetson-inference)"
OUTPUT_DIR="../data/networks"
LOG="[jetson-inference] "


#
# download a file from URL
#
function download_file()
{
	local filename=$1
	local URL=$2
	
	wget --no-check-certificate $URL -O $filename
	mv $filename $OUTPUT_DIR
}


#
# check if a particular deb package is installed with dpkg-query
# arg $1 -> package name
# arg $2 -> variable name to output status to (e.g. HAS_PACKAGE=1)
#
function find_deb_package()
{
	local PKG_NAME=$1
	local HAS_PKG=`dpkg-query -W --showformat='${Status}\n' $PKG_NAME|grep "install ok installed"`

	if [ "$HAS_PKG" == "" ]; then
		echo "$LOG Checking for '$PKG_NAME' deb package...not installed"
	else
		echo "$LOG Checking for '$PKG_NAME' deb package...installed"
		eval "$2=INSTALLED"
	fi
}


#
# install a debian package if it isn't already installed
# arg $1 -> package name
# arg $2 -> variable name to output status to (e.g. FOUND_PACKAGE=INSTALLED)
#
function install_deb_package()
{
	local PKG_NAME=$1
	
	# check to see if the package is already installed
	find_deb_package $PKG_NAME $2

	# if not, install the package
	if [ -z $2 ]; then
		echo "$LOG Missing '$PKG_NAME' deb package...installing '$PKG_NAME' package."
		sudo apt-get --force-yes --yes install $PKG_NAME
	else
		return 0
	fi
	
	# verify that the package was installed
	find_deb_package $PKG_NAME $2
	
	if [ -z $2 ]; then
		echo "$LOG Failed to install '$PKG_NAME' deb package."
		return 1
	else
		echo "$LOG Successfully installed '$PKG_NAME' deb package."
		return 0
	fi
}


# check for dialog package
install_deb_package "dialog" FOUND_DIALOG
echo "$LOG FOUND_DIALOG=$FOUND_DIALOG"

# use customized RC config
export DIALOGRC=./download-models.rc


#
# main menu
#
while true; do

	pkg_selected=$(dialog --backtitle "$APP_TITLE" \
							  --title "Model Downloader" \
							  --checklist "Keys:\n  ↑↓  Navigate Menu\n  Space to Select Models \n  Enter to Continue" 20 80 10 \
							  --output-fd 1 \
							  1 "Alexnet" off \
							  2 "Googlenet" on \
							  3 "Googlenet-12" off \
							  4 "SSD-Mobilenet-v1" off \
							  5 "SSD-Mobilenet-v2" on \
							  6 "SSD-Inception-v2" off \
							  7 "PedNet" on \
							  8 "MultiPed" off \
							  9 "FaceNet" on \
							  10 "DetectNet-COCO-Dog" on )

	pkg_selection_status=$?
	#clear

	echo "$LOG Packages selection status:  $pkg_selection_status"

	if [ $pkg_selection_status = 0 ]; then
		if [ -z $pkg_selected ]; then
			echo "$LOG No packages were selected for installation."
		else
		    echo "$LOG Packages selected for installation:  $pkg_selected"
		
			for pkg in $pkg_selected
			do
				if [ $pkg = 1 ]; then
					echo "$LOG Downloading Alexnet..."
					#install_jetson_inference
				elif [ $pkg = 2 ]; then
					echo "$LOG Downloading GoogleNet..."
					download_file "bvlc_googlenet.caffemodel" "https://nvidia.box.com/shared/static/at8b1105ww1c5h7p30j5ko8qfnxrs0eg.caffemodel" 
				elif [ $pkg = 3 ]; then
					echo "$LOG Downloading GoogleNet-12..."
				elif [ $pkg = 4 ]; then
					echo "$LOG Downloading SSD-Mobilenet-v1..."
				elif [ $pkg = 5 ]; then
					echo "$LOG Downloading SSD-Mobilenet-v2..."
				elif [ $pkg = 6 ]; then
					echo "$LOG Downloading SSD-Inception-v2..."
				elif [ $pkg = 7 ]; then
					echo "$LOG Downloading PedNet..."
				fi
			done
		fi

		exit 0
	else
		echo "$LOG Package selection cancelled."
		exit 0
	fi

	echo "$LOG Press Enter key to quit."
done

