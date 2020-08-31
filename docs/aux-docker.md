<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="aux-image.md">Back</a> | <a href="../README.md#hello-ai-world">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Appendix</sup></p>  

# Running in Docker Containers

Docker container images for this project are hosted at [`dustynv/jetson-inference`](https://hub.docker.com/r/dustynv/jetson-inference).  

Below are the currently available tags:

| Container Tag                                                                           | L4T version |          JetPack version         |
|-----------------------------------------------------------------------------------------|:-----------:|:--------------------------------:|
| [`dustynv/jetson-inference:r32.4.3`](https://hub.docker.com/r/dustynv/jetson-inference) | L4T R32.4.3 | JetPack 4.4 (production release) |

> ***note:*** the version of JetPack-L4T that you have installed on your Jetson needs to match the tag above.

These containers use the l4t-pytorch base container, so support for transfer learning / re-training is already included.

## Launching the Container

Due to various mounts and devices needed to run the container, it's recommended to use the [`docker/run.sh`](../docker/run.sh) script to run the jetson-inference container:

```bash
$ git clone --recursive https://github.com/dusty-nv/jetson-inference
$ cd jetson-inference
$ docker/run.sh
```

> ***note:***  because of the Docker scripts used and the data directory structure that gets mounted into the container, you should still clone the project on your host device (i.e. even if not intending to build/install the project natively)

[`docker/run.sh`](../docker/run.sh) will automatically pull the correct container tag from DockerHub based on your currently-installed version of JetPack-L4T, and mount the appropriate data directories and devices so that you can use cameras/display/ect from within the container.  It will also prompt you to download DNN models if you haven't already done so, which get mounted into the container to use.

### Mounted Data Volumes

For reference, the following paths automatically get mounted from your host device into the container:

* `jetson-inference/data` (stores the network models, serialized TensorRT engines, and test images)
* `jetson-inference/python/training/classification/data` (stores classification training datasets)
* `jetson-inference/python/training/classification/models` (stores classification models trained by PyTorch)
* `jetson-inference/python/training/detection/ssd/data` (stores detection training datasets)
* `jetson-inference/python/training/detection/ssd/models` (stores detection models trained by PyTorch)

These mounted volumes assure that the models and datasets are stored outside the container, and aren't lost when the container is shut down.

If you wish to mount your own path into the container, you can use the `--volume HOST_DIR:MOUNT_DIR` argument to [`docker/run.sh`](../docker/run.sh):

```bash
$ docker/run.sh --volume /my/host/path:/my/container/path
```

For more info, see `docker/run.sh --help`:

```bash
   -v, --volume HOST_DIR:MOUNT_DIR Mount a path from the host system into
                                   the container.  Should be specified as:

                                      -v /my/host/path:/my/container/path

                                   (these should be absolute paths)
```

## Running Applications

Once the container is running, you can run example programs from the tutorial like normal:

```bash
# cd build/aarch64/bin
# ./video-viewer /dev/video0
# ./imagenet images/jellyfish.jpg images/test/jellyfish.jpg
# ./detectnet images/peds_0.jpg images/test/peds_0.jpg
```

## Building the Container

If you wish to re-build the container or build your own, you can use the [`docker/build.sh`](../docker/build.sh) script which builds the project's [`Dockerfile`](../Dockerfile):

```bash
$ docker/build.sh
```

>  ***note:*** first you should set your default `docker-runtime` to nvidia, see [here](https://github.com/dusty-nv/jetson-containers#docker-default-runtime) for the details.


##
<p align="right">Back | <b><a href="aux-image.md">Image Manipulation with CUDA</a></p>
<p align="center"><sup>© 2016-2020 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>

