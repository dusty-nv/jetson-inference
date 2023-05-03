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
LOG="[jetson-inference] "
WGET_QUIET="--quiet"
BUILD_INTERACTIVE=${1:-"YES"}


#
# exit message for user
#
function exit_message()
{
	echo " "

	if [ $1 = 0 ]; then
		echo "$LOG installation complete, exiting with status code $1"
	else
		echo "$LOG errors encountered during installation, exiting with code $1"
	fi

	echo "$LOG to run this tool again, use the following commands:"
	echo " "
	echo "    $ cd <jetson-inference>/build"
	echo "    $ ./install-pytorch.sh"
	echo " "

	exit $1
}

#
# prompt user for retry
#
function retry_prompt()
{
	dialog --backtitle "$APP_TITLE" \
			  --title "Download Error" \
			  --colors \
			  --extra-button \
			  --extra-label "Next" \
			  --cancel-label "Quit" \
			  --ok-label "Retry" \
			  --yesno "\nFailed to download '$1' (error code=$2)\n\nWould you like to try downloading it again?\n\n\ZbNote:\Zn  if this error keeps occuring, see here:\n https://eLinux.org/Jetson_Zoo" 12 60

	local retry_status=$?
	clear

	WGET_QUIET="--verbose"

	if [ $retry_status = 1 ]; then
		echo "$LOG packages failed to download"
		exit_message 1
	elif [ $retry_status != 0 ]; then
		return 1
	fi

	return 0
}


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

	#mv $filename $OUTPUT_DIR
	return 0
}


#
# download a file from URL
#
function download_file()
{
	local filename=$1
	local URL=$2
	
	WGET_QUIET="--quiet"

	while true; do
		attempt_download_file $filename $URL

		local download_status=$?

		if [ $download_status = 0 ]; then
			return 0
		fi

		retry_prompt $filename $download_status
	
		local retry_status=$?

		if [ $retry_status != 0 ]; then
			return 1
		fi
	done
}


#
# download and install a pip wheel
#
function download_wheel()
{
	local filename=$2
	local URL=$3
	local sudo=${4:-""}
	
	download_file $filename $URL

	local download_status=$?

	if [ $download_status != 0 ]; then
		echo "$LOG failed to download $filename"
		return 1
	fi

	$sudo $1 install $filename

	local install_status=$?

	if [ $install_status != 0 ]; then
		echo "$LOG failed to install $filename"
		echo "$LOG    -- command:     $1 install $filename"
		echo "$LOG    -- error code:  $install_status"
		return 1
	fi

	return 0
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
		return 1
	else
		echo "$LOG Checking for '$PKG_NAME' deb package...installed"
		eval "$2=INSTALLED"
		return 0
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

	local pkg_status=$?

	# if not, install the package
	if [ $pkg_status != 0 ]; then
		echo "$LOG Missing '$PKG_NAME' deb package...installing '$PKG_NAME' package."
		sudo apt-get --force-yes --yes install $PKG_NAME
	else
		return 0
	fi
	
	# verify that the package was installed
	find_deb_package $PKG_NAME $2
	
	local install_status=$?

	if [ $install_status != 0 ]; then
		echo "$LOG Failed to install '$PKG_NAME' deb package."
		exit_message 1
		#return 1
	else
		echo "$LOG Successfully installed '$PKG_NAME' deb package."
		return 0
	fi
}


#
# move/restore ffmpeg
# (this is so torchvision doesn't inadvertently try to build with it)
#
function move_ffmpeg()
{
	if [ -f "/usr/bin/ffmpeg" ]; then
		echo "$LOG temporarily moving /usr/bin/ffmpeg -> /usr/bin/ffmpeg_bak"
		sudo mv /usr/bin/ffmpeg /usr/bin/ffmpeg_bak
	fi
}

function restore_ffmpeg()
{
	if [ -f "/usr/bin/ffmpeg_bak" ]; then
		echo "$LOG restoring /usr/bin/ffmpeg from /usr/bin/ffmpeg_bak"
		sudo mv /usr/bin/ffmpeg_bak /usr/bin/ffmpeg
	fi
}


#
# install PyTorch
#
function install_pytorch()
{
	local pytorch_version=$1
	local python_version=$2
     local l4t_release=$3
	local l4t_revision=$4
	
	if [ $pytorch_version = "1.1.0" ]; then
	
		if [ $python_version == "2.7" ]; then
			install_pytorch_v110_python27_jp42
		elif [ $python_version == "3.6" ]; then
			install_pytorch_v110_python36_jp42
		fi
		
	elif [ $pytorch_version = "1.4.0" ]; then
			
		if [ $JETSON_L4T_RELEASE -eq 32 ]; then
			if [ $JETSON_L4T_REVISION = "4.2" ]; then
				if [ $python_version == "2.7" ]; then	# JetPack 4.4 DP
					install_pytorch_v140_python27_jp44
				elif [ $python_version == "3.6" ]; then
					install_pytorch_v140_python36_jp44
				fi
			else
				if [ $python_version == "2.7" ]; then	# JetPack 4.2, 4.3
					install_pytorch_v140_python27_jp42
				elif [ $python_version == "3.6" ]; then
					install_pytorch_v140_python36_jp42
				fi
			fi
		fi
		
	elif [ $pytorch_version = "1.6.0" ]; then
		install_pytorch_v160_python36_jp44
	elif [ $pytorch_version = "1.12" ]; then
		install_pytorch_v1120_python38_jp50
	elif [ $pytorch_version = "2.0" ]; then
		install_pytorch_v200_python38_jp50
	else
		echo "$LOG invalid PyTorch version selected:  PyTorch $pytorch_version"
		exit_message 1
	fi

	return $?
}
	
function install_pytorch_v110_python27_jp42()
{
	echo "$LOG Downloading PyTorch v1.1.0 (Python 2.7)..."

	# install apt packages
	install_deb_package "python-pip" FOUND_PIP
	install_deb_package "qtbase5-dev" FOUND_QT5
	install_deb_package "libjpeg-dev" FOUND_JPEG
	install_deb_package "zlib1g-dev" FOUND_ZLIB

	# install pytorch wheel
	download_wheel pip "torch-1.1.0-cp27-cp27mu-linux_aarch64.whl" "https://nvidia.box.com/shared/static/o8teczquxgul2vjukwd4p77c6869xmri.whl"

	local wheel_status=$?

	if [ $wheel_status != 0 ]; then
		echo "$LOG failed to install PyTorch v1.1.0 (Python 2.7)"
		return 1
	fi

	# build torchvision
	move_ffmpeg
	echo "$LOG cloning torchvision..."
	sudo rm -r -f torchvision-27
	git clone -bv0.3.0 https://github.com/dusty-nv/vision torchvision-27
	cd torchvision-27
	echo "$LOG building torchvision for Python 2.7..."
	sudo python setup.py install
	cd ../
	restore_ffmpeg

	# patch for https://github.com/pytorch/vision/issues/1712
	pip install 'pillow<7'
	
	return 0
}

function install_pytorch_v110_python36_jp42()
{
	echo "$LOG Downloading PyTorch v1.1.0 (Python 3.6)..."

	# install apt packages
	install_deb_package "python3-pip" FOUND_PIP3
	install_deb_package "qtbase5-dev" FOUND_QT5
	install_deb_package "libjpeg-dev" FOUND_JPEG
	install_deb_package "zlib1g-dev" FOUND_ZLIB

	# install pytorch wheel
	download_wheel pip3 "torch-1.1.0-cp36-cp36m-linux_aarch64.whl" "https://nvidia.box.com/shared/static/j2dn48btaxosqp0zremqqm8pjelriyvs.whl"

	local wheel_status=$?

	if [ $wheel_status != 0 ]; then
		echo "$LOG failed to install PyTorch v1.1.0 (Python 3.6)"
		return 1
	fi

	# build torchvision
	move_ffmpeg
	echo "$LOG cloning torchvision..."
	sudo rm -r -f torchvision-36
	git clone -bv0.3.0 https://github.com/dusty-nv/vision torchvision-36
	cd torchvision-36
	echo "$LOG building torchvision for Python 3.6..."
	sudo python3 setup.py install
	cd ../
	restore_ffmpeg

	# patch for https://github.com/pytorch/vision/issues/1712
	pip3 install 'pillow<7'
	
	return 0
}

function install_pytorch_v140_python27_jp42()
{
	echo "$LOG Downloading PyTorch v1.4.0 (Python 2.7)..."

	# install apt packages
	install_deb_package "python-pip" FOUND_PIP
	install_deb_package "qtbase5-dev" FOUND_QT5
	install_deb_package "libjpeg-dev" FOUND_JPEG
	install_deb_package "zlib1g-dev" FOUND_ZLIB
	install_deb_package "libopenblas-base" FOUND_OPENBLAS
	install_deb_package "libopenmpi-dev" FOUND_OPENMPI

	# install pip packages
	pip install future

	# install pytorch wheel
	download_wheel pip "torch-1.4.0-cp27-cp27mu-linux_aarch64.whl" "https://nvidia.box.com/shared/static/1v2cc4ro6zvsbu0p8h6qcuaqco1qcsif.whl"

	local wheel_status=$?

	if [ $wheel_status != 0 ]; then
		echo "$LOG failed to install PyTorch v1.4.0 (Python 2.7)"
		return 1
	fi

	# patch for https://github.com/python-pillow/Pillow/issues/4478
	pip install 'pillow<7'

	# build torchvision
	move_ffmpeg
	echo "$LOG cloning torchvision..."
	sudo rm -r -f torchvision-27
	git clone -bv0.5.0 https://github.com/pytorch/vision torchvision-27
	cd torchvision-27
	echo "$LOG building torchvision for Python 2.7..."
	sudo python setup.py install
	cd ../
	restore_ffmpeg
	
	return 0
}

function install_pytorch_v140_python36_jp42()
{
	echo "$LOG Downloading PyTorch v1.4.0 (Python 3.6)..."

	# install apt packages
	install_deb_package "python3-pip" FOUND_PIP3
	install_deb_package "qtbase5-dev" FOUND_QT5
	install_deb_package "libjpeg-dev" FOUND_JPEG
	install_deb_package "zlib1g-dev" FOUND_ZLIB
	install_deb_package "libopenblas-base" FOUND_OPENBLAS
	install_deb_package "libopenmpi-dev" FOUND_OPENMPI

	# install pip packages
	pip3 install Cython
	pip3 install numpy --verbose

	# install pytorch wheel
	download_wheel pip3 "torch-1.4.0-cp36-cp36m-linux_aarch64.whl" "https://nvidia.box.com/shared/static/ncgzus5o23uck9i5oth2n8n06k340l6k.whl"

	local wheel_status=$?

	if [ $wheel_status != 0 ]; then
		echo "$LOG failed to install PyTorch v1.4.0 (Python 3.6)"
		return 1
	fi

	# build torchvision
	move_ffmpeg
	echo "$LOG cloning torchvision..."
	sudo rm -r -f torchvision-36
	git clone -bv0.5.0 https://github.com/pytorch/vision torchvision-36
	cd torchvision-36
	echo "$LOG building torchvision for Python 3.6..."
	sudo python3 setup.py install
	cd ../
	restore_ffmpeg

	return 0
}

function install_pytorch_v140_python27_jp44()
{
	echo "$LOG Downloading PyTorch v1.4.0 (Python 2.7)..."

	# install apt packages
	install_deb_package "python-pip" FOUND_PIP
	install_deb_package "qtbase5-dev" FOUND_QT5
	install_deb_package "libjpeg-dev" FOUND_JPEG
	install_deb_package "zlib1g-dev" FOUND_ZLIB
	install_deb_package "libopenblas-base" FOUND_OPENBLAS
	install_deb_package "libopenmpi-dev" FOUND_OPENMPI

	# install pip packages
	pip install future

	# install pytorch wheel
	download_wheel pip "torch-1.4.0-cp27-cp27mu-linux_aarch64.whl" "https://nvidia.box.com/shared/static/yhlmaie35hu8jv2xzvtxsh0rrpcu97yj.whl"

	local wheel_status=$?

	if [ $wheel_status != 0 ]; then
		echo "$LOG failed to install PyTorch v1.4.0 (Python 2.7)"
		return 1
	fi

	# patch for https://github.com/python-pillow/Pillow/issues/4478
	pip install 'pillow<7'

	# build torchvision
	move_ffmpeg
	echo "$LOG cloning torchvision..."
	sudo rm -r -f torchvision-27
	git clone -bv0.5.0 https://github.com/pytorch/vision torchvision-27
	cd torchvision-27
	echo "$LOG building torchvision for Python 2.7..."
	sudo python setup.py install
	cd ../
	restore_ffmpeg

	return 0
}

function install_pytorch_v140_python36_jp44()
{
	echo "$LOG Downloading PyTorch v1.4.0 (Python 3.6)..."

	# install apt packages
	install_deb_package "python3-pip" FOUND_PIP3
	install_deb_package "qtbase5-dev" FOUND_QT5
	install_deb_package "libjpeg-dev" FOUND_JPEG
	install_deb_package "zlib1g-dev" FOUND_ZLIB
	install_deb_package "libopenblas-base" FOUND_OPENBLAS
	install_deb_package "libopenmpi-dev" FOUND_OPENMPI

	# install pip packages
	pip3 install Cython
	pip3 install numpy --verbose
	pip3 install tensorboard --verbose
	
	# install pytorch wheel
	download_wheel pip3 "torch-1.4.0-cp36-cp36m-linux_aarch64.whl" "https://nvidia.box.com/shared/static/c3d7vm4gcs9m728j6o5vjay2jdedqb55.whl"

	local wheel_status=$?

	if [ $wheel_status != 0 ]; then
		echo "$LOG failed to install PyTorch v1.4.0 (Python 3.6)"
		return 1
	fi

	# build torchvision
	move_ffmpeg
	echo "$LOG cloning torchvision..."
	sudo rm -r -f torchvision-36
	git clone -bv0.5.0 https://github.com/pytorch/vision torchvision-36
	cd torchvision-36
	echo "$LOG building torchvision for Python 3.6..."
	sudo python3 setup.py install
	cd ../
	restore_ffmpeg

	return 0
}

function install_pytorch_v160_python36_jp44()
{
	echo "$LOG Downloading PyTorch v1.6.0 (Python 3.6)..."

	# install apt packages
	install_deb_package "python3-pip" FOUND_PIP3
	install_deb_package "qtbase5-dev" FOUND_QT5
	install_deb_package "libjpeg-dev" FOUND_JPEG
	install_deb_package "zlib1g-dev" FOUND_ZLIB
	install_deb_package "libopenblas-base" FOUND_OPENBLAS
	install_deb_package "libopenmpi-dev" FOUND_OPENMPI

	# install pip packages
	pip3 install Cython
	pip3 install numpy --verbose
	pip3 install tensorboard --verbose
	
	# install pytorch wheel
	download_wheel pip3 "torch-1.6.0-cp36-cp36m-linux_aarch64.whl" "https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl"

	local wheel_status=$?

	if [ $wheel_status != 0 ]; then
		echo "$LOG failed to install PyTorch v1.6.0 (Python 3.6)"
		return 1
	fi

	# build torchvision
	move_ffmpeg
	echo "$LOG cloning torchvision..."
	sudo rm -r -f torchvision-36
	git clone -bv0.7.0 https://github.com/pytorch/vision torchvision-36
	cd torchvision-36
	echo "$LOG building torchvision for Python 3.6..."
	sudo python3 setup.py install
	cd ../
	restore_ffmpeg

	return 0
}

function install_pytorch_v1120_python38_jp50()
{
	echo "$LOG Downloading PyTorch v1.12.0 (Python 3.8)..."

	# install apt packages
	install_deb_package "python3-pip" FOUND_PIP3
	install_deb_package "qtbase5-dev" FOUND_QT5
	install_deb_package "libjpeg-dev" FOUND_JPEG
	install_deb_package "zlib1g-dev" FOUND_ZLIB
	install_deb_package "libopenblas-base" FOUND_OPENBLAS
	install_deb_package "libopenmpi-dev" FOUND_OPENMPI
	install_deb_package "libomp-dev" FOUND_OPENMP
	install_deb_package "ninja-build" FOUND_NINJA
	
	# install pip packages
	pip3 install Cython
	pip3 install numpy --verbose
	pip3 install tensorboard --verbose
	
	# install pytorch wheel
	download_wheel pip3 "torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl" "https://developer.download.nvidia.com/compute/redist/jp/v50/pytorch/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl" "sudo"

	local wheel_status=$?

	if [ $wheel_status != 0 ]; then
		echo "$LOG failed to install PyTorch v1.12.0 (Python 3.8)"
		return 1
	fi

	# build torchvision
	move_ffmpeg
	echo "$LOG cloning torchvision..."
	sudo rm -r -f torchvision-38
	git clone -bv0.12.0 --depth=1 https://github.com/pytorch/vision torchvision-38
	cd torchvision-38
	echo "$LOG building torchvision for Python 3.8..."
	sudo python3 setup.py install
	cd ../
	restore_ffmpeg

	return 0
}

function install_pytorch_v200_python38_jp50()
{
	echo "$LOG Downloading PyTorch v2.0 (Python 3.8)..."

	# install apt packages
	install_deb_package "python3-pip" FOUND_PIP3
	install_deb_package "qtbase5-dev" FOUND_QT5
	install_deb_package "libjpeg-dev" FOUND_JPEG
	install_deb_package "zlib1g-dev" FOUND_ZLIB
	install_deb_package "libopenblas-base" FOUND_OPENBLAS
	install_deb_package "libopenmpi-dev" FOUND_OPENMPI
	install_deb_package "libomp-dev" FOUND_OPENMP
	install_deb_package "ninja-build" FOUND_NINJA
	
	# install pip packages
	pip3 install Cython
	pip3 install numpy --verbose
	pip3 install tensorboard --verbose
	pip3 install onnx --verbose
	
	# install pytorch wheel
	download_wheel pip3 "torch-2.0.0.nv23.05-cp38-cp38-linux_aarch64.whl" "https://nvidia.box.com/shared/static/sct3njlmea4whlf6ud9tj1853zi3vb1v.whl" "sudo"

	local wheel_status=$?

	if [ $wheel_status != 0 ]; then
		echo "$LOG failed to install PyTorch v2.0 (Python 3.8)"
		return 1
	fi

	# build torchvision
	move_ffmpeg
	echo "$LOG cloning torchvision..."
	sudo rm -r -f torchvision-38
	git clone -bv0.15.1 --depth=1 https://github.com/pytorch/vision torchvision-38
	cd torchvision-38
	echo "$LOG building torchvision for Python 3.8..."
	sudo python3 setup.py install
	cd ../
	restore_ffmpeg

	return 0
}

#
# check L4T version
#
function check_L4T_version()
{
	JETSON_L4T_STRING=$(head -n 1 /etc/nv_tegra_release)

	if [ -z $JETSON_L4T_STRING ]; then
		echo "$LOG reading L4T version from \"dpkg-query --show nvidia-l4t-core\""

		JETSON_L4T_STRING=$(dpkg-query --showformat='${Version}' --show nvidia-l4t-core)
		local JETSON_L4T_ARRAY=(${JETSON_L4T_STRING//./ })	

		#echo ${JETSON_L4T_ARRAY[@]}
		#echo ${#JETSON_L4T_ARRAY[@]}

		JETSON_L4T_RELEASE=${JETSON_L4T_ARRAY[0]}
		JETSON_L4T_REVISION=${JETSON_L4T_ARRAY[1]}
	else
		echo "$LOG reading L4T version from /etc/nv_tegra_release"

		JETSON_L4T_RELEASE=$(echo $JETSON_L4T_STRING | cut -f 2 -d ' ' | grep -Po '(?<=R)[^;]+')
		JETSON_L4T_REVISION=$(echo $JETSON_L4T_STRING | cut -f 2 -d ',' | grep -Po '(?<=REVISION: )[^;]+')
	fi

	JETSON_L4T_REVISION_MAJOR=${JETSON_L4T_REVISION:0:1}
	JETSON_L4T_REVISION_MINOR=${JETSON_L4T_REVISION:2:1}

	JETSON_L4T_VERSION="$JETSON_L4T_RELEASE.$JETSON_L4T_REVISION"
	echo "$LOG Jetson BSP Version:  L4T R$JETSON_L4T_VERSION"

	if [ $JETSON_L4T_RELEASE -lt 32 ]; then
		dialog --backtitle "$APP_TITLE" \
		  --title "PyTorch Automated Install requires JetPack ≥4.2" \
		  --colors \
		  --msgbox "\nThis script to install PyTorch from pre-built binaries\nrequires \ZbJetPack 4.2 or newer\Zn (L4T R32.1 or newer).\n\nThe version of L4T on your system is:  \ZbL4T R${JETSON_L4T_VERSION}\Zn\n\nIf you wish to install PyTorch for training on Jetson,\nplease upgrade to JetPack 4.2 or newer, or see these\ninstructions to build PyTorch from source:\n\n          \Zbhttps://eLinux.org/Jetson_Zoo\Zn\n\nNote that PyTorch isn't required to build the repo,\njust for re-training networks onboard your Jetson.\nYou can proceed following Hello AI World without it,\nexcept for the parts on Transfer Learning with PyTorch." 20 60

		clear
		echo " "
		echo "[jetson-inference]  this script to install PyTorch from pre-built binaries"
		echo "                    requires JetPack 4.2 or newer (L4T R32.1 or newer).  "
		echo "                    the version of L4T on your system is:  L4T R${JETSON_L4T_VERSION}"
		echo " "
		echo "                    if you wish to install PyTorch for training on Jetson,"
		echo "                    please upgrade to JetPack 4.2 or newer, or see these"
		echo "                    instructions to build PyTorch from source:"
		echo " "
		echo "                        > https://eLinux.org/Jetson_Zoo"
		echo " "
		echo "                    note that PyTorch isn't required to build the repo,"
		echo "                    just for re-training networks onboard your Jetson."
		echo " "
		echo "                    you can proceed following Hello AI World without it,"
		echo "                    except for the parts on Transfer Learning with PyTorch."

		exit_message 1
	fi
}


#
# non-interactive mode
#
echo "$LOG BUILD_INTERACTVE=$BUILD_INTERACTIVE"

if [[ "$BUILD_INTERACTIVE" != "YES" ]]; then
	echo "$LOG non-interactive mode, skipping PyTorch install..."
	exit_message 0
fi


# check for dialog package
install_deb_package "dialog" FOUND_DIALOG
echo "$LOG FOUND_DIALOG=$FOUND_DIALOG"

# use customized RC config
export DIALOGRC=./install-pytorch.rc

# check L4T version
check_L4T_version


#
# main menu
#
while true; do

	HAS_PYTHON2=false
	PYTHON3_VERSION="3.6"
	PYTHON_VERSION_ONE=$PYTHON3_VERSION
		
	if [ $JETSON_L4T_RELEASE -eq 32 ]; then
		if [ $JETSON_L4T_REVISION = "4.3" ] || [ $JETSON_L4T_REVISION_MAJOR -gt 4 ]; then
			PYTORCH_VERSION="1.6.0"  # JetPack 4.4 GA
		elif [ $JETSON_L4T_REVISION_MAJOR -eq 4 ] && [ $JETSON_L4T_REVISION_MINOR -ge 3 ]; then
			PYTORCH_VERSION="1.6.0"
		elif [ $JETSON_L4T_REVISION = "4.2" ]; then
			PYTORCH_VERSION="1.4.0"	# JetPack 4.4 DP
			HAS_PYTHON2=true
		else
			PYTORCH_VERSION="1.4.0"	# JetPack 4.2, 4.3
			HAS_PYTHON2=true
		fi
	elif [[ $JETSON_L4T_RELEASE -eq 34 || $JETSON_L4T_RELEASE -eq 35 ]]; then
		# JetPack 5.x
		PYTHON3_VERSION="3.8"
		PYTORCH_VERSION="2.0"
		PYTORCH_VERSION_TWO="1.12"
		PYTHON_VERSION_ONE=$PYTHON3_VERSION
	fi

     if [ "$HAS_PYTHON2" = true ]; then
		PYTHON_VERSION_ONE="2.7"
		PYTHON_VERSION_TWO=$PYTHON3_VERSION
		
		packages_selected=$(dialog --backtitle "$APP_TITLE" \
						  --title "PyTorch Installer (L4T R$JETSON_L4T_VERSION)" \
						  --cancel-label "Skip" \
						  --colors \
						  --checklist "If you want to train DNN models on your Jetson, this tool will download and install PyTorch.  Select the desired versions of pre-built packages below, or see \Zbhttp://eLinux.org/Jetson_Zoo\Zn for instructions to build from source. \n\nYou can skip this step and select Skip if you don't want to install PyTorch.\n\n\ZbKeys:\Zn\n  ↑↓ Navigate Menu\n  Space to Select \n  Enter to Continue\n\n\ZbPackages to Install:\Zn" 20 80 2 \
						  --output-fd 1 \
						  1 "PyTorch $PYTORCH_VERSION for Python $PYTHON_VERSION_ONE" off \
						  2 "PyTorch $PYTORCH_VERSION for Python $PYTHON_VERSION_TWO" off \
						 )
	elif [ -n "$PYTORCH_VERSION_TWO" ]; then
		packages_selected=$(dialog --backtitle "$APP_TITLE" \
						  --title "PyTorch Installer (L4T R$JETSON_L4T_VERSION)" \
						  --cancel-label "Skip" \
						  --colors \
						  --checklist "If you want to train DNN models on your Jetson, this tool will download and install PyTorch.  Select the desired versions of pre-built packages below, or see \Zbhttp://eLinux.org/Jetson_Zoo\Zn for instructions to build from source. \n\nYou can skip this step and select Skip if you don't want to install PyTorch.\n\n\ZbKeys:\Zn\n  ↑↓ Navigate Menu\n  Space to Select \n  Enter to Continue\n\n\ZbPackages to Install:\Zn" 20 80 2 \
						  --output-fd 1 \
						  1 "PyTorch $PYTORCH_VERSION for Python $PYTHON_VERSION_ONE" off \
						  2 "PyTorch $PYTORCH_VERSION_TWO for Python $PYTHON_VERSION_ONE" off \
						 )
	else
		packages_selected=$(dialog --backtitle "$APP_TITLE" \
						  --title "PyTorch Installer (L4T R$JETSON_L4T_VERSION)" \
						  --cancel-label "Skip" \
						  --colors \
						  --checklist "If you want to train DNN models on your Jetson, this tool will download and install PyTorch.  Select the desired versions of pre-built packages below, or see \Zbhttp://eLinux.org/Jetson_Zoo\Zn for instructions to build from source. \n\nYou can skip this step and select Skip if you don't want to install PyTorch.\n\n\ZbKeys:\Zn\n  ↑↓ Navigate Menu\n  Space to Select \n  Enter to Continue\n\n\ZbPackages to Install:\Zn" 20 80 1 \
						  --output-fd 1 \
						  1 "PyTorch $PYTORCH_VERSION for Python $PYTHON_VERSION_ONE" off \
						 )
	fi
	
	if [ -z "$PYTORCH_VERSION_TWO" ]; then
		PYTORCH_VERSION_TWO=$PYTORCH_VERSION
	fi 
	
	package_selection_status=$?
	clear

	echo "$LOG Package selection status:  $package_selection_status"

	if [ $package_selection_status = 0 ]; then

		if [ -z "$packages_selected" ]; then
			echo "$LOG No packages were selected for download."
		else
			echo "$LOG Packages selected for download:  $packages_selected"
		
			for pkg in $packages_selected
			do
				if [ $pkg = 1 ]; then
					install_pytorch $PYTORCH_VERSION $PYTHON_VERSION_ONE $JETSON_L4T_RELEASE $JETSON_L4T_REVISION
				elif [ $pkg = 2 ]; then
					install_pytorch $PYTORCH_VERSION_TWO $PYTHON_VERSION_TWO $JETSON_L4T_RELEASE $JETSON_L4T_REVISION
				fi
			done
		fi

		exit_message 0
	else
		echo "$LOG Package selection cancelled."
		exit_message 0
	fi

	exit_message 0
done

