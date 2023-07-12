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

ARCH=$(uname -i)
echo "ARCH:  $ARCH"

if [ $ARCH = "aarch64" ]; then
	L4T_VERSION_STRING=$(head -n 1 /etc/nv_tegra_release)

	if [ -z "$L4T_VERSION_STRING" ]; then
		echo "reading L4T version from \"dpkg-query --show nvidia-l4t-core\""

		L4T_VERSION_STRING=$(dpkg-query --showformat='${Version}' --show nvidia-l4t-core)
		L4T_VERSION_ARRAY=(${L4T_VERSION_STRING//./ })	

		#echo ${L4T_VERSION_ARRAY[@]}
		#echo ${#L4T_VERSION_ARRAY[@]}

		L4T_RELEASE=${L4T_VERSION_ARRAY[0]}
		L4T_REVISION=${L4T_VERSION_ARRAY[1]}
	else
		echo "reading L4T version from /etc/nv_tegra_release"

		L4T_RELEASE=$(echo $L4T_VERSION_STRING | cut -f 2 -d ' ' | grep -Po '(?<=R)[^;]+')
		L4T_REVISION=$(echo $L4T_VERSION_STRING | cut -f 2 -d ',' | grep -Po '(?<=REVISION: )[^;]+')
	fi

	L4T_REVISION_MAJOR=${L4T_REVISION:0:1}
	L4T_REVISION_MINOR=${L4T_REVISION:2:1}

	L4T_VERSION="$L4T_RELEASE.$L4T_REVISION"

	echo "L4T BSP Version:  L4T R$L4T_VERSION"
	
elif [ $ARCH != "x86_64" ]; then
	echo "unsupported architecture:  $ARCH"
	exit 1
fi
