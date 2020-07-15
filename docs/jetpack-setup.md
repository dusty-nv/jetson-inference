<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="digits-setup.md">Back</a> | <a href="building-repo.md">Next</a> | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Contents</sup></a>
<br/>
<sup>System Setup</sup></p> 

# Setting up Jetson with JetPack

> **note**:  if your Jetson Nano has already been setup with the [SD card image](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write) (which includes the JetPack components), or your Jetson has already been setup with JetPack, you can skip this step and continue to [`Building the Repo`](building-repo.md)

Download the latest **[JetPack](https://developer.nvidia.com/embedded/jetpack)** to your host PC.  In addition to flashing the Jetson with the latest Board Support Package (BSP), JetPack automatically installs tools for the host like CUDA Toolkit.  See the JetPack [Release Notes](https://developer.nvidia.com/embedded/jetpack-notes) for the full list of features and installed packages.

After downloading JetPack from the link above, run it from the host PC with the following commands:

``` bash 
$ cd <directory where you downloaded JetPack>
$ chmod +x JetPack-L4T-<version>-linux-x64.run 
$ ./JetPack-L4T-<version>-linux-x64.run 
```

The JetPack GUI will start.  Follow the step-by-step **[Install Guide](http://docs.nvidia.com/jetpack-l4t/index.html#developertools/mobile/jetpack/l4t/3.0/jetpack_l4t_install.htm)** to complete the setup.  Near the beginning, JetPack will confirm which generation Jetson you are developing for.

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/jetpack-platform.png" width="450">

Select Jetson TX1 if you are using TX1, or Jetson TX2 if you're using TX2, and press `Next` to continue.

The next screen will list the packages available to be installed.  The packages installed to the host are listed at the top under the `Host - Ubuntu` dropdown, while those intended for the Jetson are shown near the bottom.  You can select or deselect an individual package for installation by clicking it's `Action` column.

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/jetpack-downloads.png" width="500">

Since CUDA will be used on the host for training DNNs, it's recommended to select the Full install by click on the radio button in the top right.  Then press `Next` to begin setup.  JetPack will download and then install the sequence of packages.  Note that all the .deb packages are stored under the `jetpack_downloads` subdirectory if you are to need them later.  

After the downloads have finished installing, JetPack will enter the post-install phase where the JetPack is flashed with the L4T BSP.  You'll need to connect your Jetson to your host PC via the micro-USB port and cable included in the devkit.  Then enter your Jetson into recovery mode by holding down the Recovery button while pressing and releasing Reset.  If you type `lsusb` from the host PC after you've connected the micro-USB cable and entered the Jetson into recovery mode, you should see the NVIDIA device come up under the list of USB devices.  JetPack uses the micro-USB connection from the host to flash the L4T BSP to the Jetson.  

After flashing, the Jetson will reboot and if attached to an HDMI display, will boot up to the Ubuntu desktop.  After this, JetPack connects to the Jetson from the host via SSH to install additional packages to the Jetson, like the ARM aarch64 builds of CUDA Toolkit, cuDNN, and TensorRT.  For JetPack to be able to reach the Jetson via SSH, the host PC should be networked to the Jetson via Ethernet.  This can be accomplished by running an Ethernet cable directly from the host to the Jetson, or by connecting both devices to a router or switch — the JetPack GUI will ask you to confirm which networking scenario is being used.  

Please refer to the **[JetPack Install Guide](http://docs.nvidia.com/jetpack-l4t/index.html#developertools/mobile/jetpack/l4t/3.0/jetpack_l4t_install.htm)** for the full directions for installing JetPack and flashing Jetson.

##
<p align="right">Next | <b><a href="building-repo.md">Building the Repo from Source</a></b>
<br/>
Back | <b><a href="digits-setup.md">DIGITS System Setup</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Table of Contents</sup></a></p>
