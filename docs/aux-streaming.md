<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="../README.md#hello-ai-world">Back</a> | <a href="aux-image-manipulation.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Appendix</sup></p>  

# Camera Streaming and Multimedia

This project supports streaming video feeds and images via a variety of interfaces and protocols, including:

* MIPI CSI cameras
* V4L2 cameras
* RTP/RTSP
* Video & image files
* Sequences of images
* OpenGL displays

Streams are identified via a resource URI and accessed through the [`videoSource`](#videoSource) and [`videoOutput`](#videoOutput) APIs.  The tables below show the supported input/output protocols and example URIs for each type of stream:

### Input Streams

|                  | Protocol     | Resource URI              | Notes                                                    |
|------------------|--------------|---------------------------|----------------------------------------------------------|
| [MIPI CSI cameras](#mipi-csi-cameras) | `csi://`     | `csi://0`                 | CSI camera 0 (substitute other camera numbers for `0`)                    |
| [V4L2 cameras](#v4l2-cameras)     | `v4l2://`    | `v4l2:///dev/video0`      | V4L2 device 0 (substitute other camera numbers for `0`)                            |
| RTP stream       | `rtp://`     | `rtp://@:1234`            | localhost, port 1234 (requires additional configuration) |
| RTSP stream      | `rtsp://`    | `rtsp://<remote-ip>:1234` | Replace `<remote-ip>` with remote host's IP or hostname  |
| Video file       | `file://`    | `file://my_video.mp4`     | Supports loading MP4, MKV, AVI, FLV (see codecs below)   |
| Image file       | `file://`    | `file://my_image.jpg`     | Supports loading JPG, PNG, TGA, BMP, GIF, ect.           |
| Image sequence   | `file://`    | `file://my_directory/`    | Searches for images in alphanumeric order                |

* Supported decoder codecs:  H.264, H.265, VP8, VP9, MPEG-2, MPEG-4, MJPEG
* The `file://`, `v4l2://`, and `csi://` protocol prefixes can be omitted from the URI as shorthand

### Output Streams

|                  | Protocol     | Resource URI              | Notes                                                    |
|------------------|--------------|---------------------------|----------------------------------------------------------|
| OpenGL display   | `display://` | `display://0`             | Creates GUI window on screen 0                           |
| RTP              | `rtp://`     | `rtp://<remote-ip>:1234`  | Replace `<remote-ip>` with remote host's IP or hostname  |
| Video file       | `file://`    | `file://my_video.mp4`     | Supports saving MP4, MKV, AVI, FLV (see codecs below)    |
| Image file       | `file://`    | `file://my_image.jpg`     | Supports saving JPG, PNG, TGA, BMP                       |
| Image sequence   | `file://`    | `file://image_%i.jpg`     | `%i` is replaced by the image number in the sequence     |

* Supported encoder codecs:  H.264, H.265, VP8, VP9, MJPEG
* The `file://` protocol prefixes can be omitted from the URI as shorthand
* By default, an OpenGL display window will be created unless `--headless` is specified

## Command-Line Arguments

Each example C++ and Python application from jetson-inference (e.g. [`imagenet`](../examples/imagenet/imagenet.cpp)/[`imagenet.py`](../examples/python/imagenet.py), [`detectnet`](../examples/detectnet/detectnet.cpp)/[`detectnet.py`](../examples/python/detectnet.py), [`segnet`](../examples/segnet/segnet.cpp)/[`segnet.py`](../examples/python/segnet.py), [`video-viewer`](https://github.com/dusty-nv/jetson-utils/video/video-viewer/video-viewer.cpp)/[`video-viewer.py`](https://github.com/dusty-nv/jetson-utils/python/examples/video-viewer.py), ect) accepts the same set of command-line arguments for specifying stream URIs and additional options.  These generally take the form:

```bash
imagenet [options] input_URI [output_URI]
```

where the input and output URIs are specified by two positional arguments.  For example:

```bash
imagenet input.jpg output.jpg
```

As mentioned above, any of the examples from jetson-inference can be substituted here, since they use the same command-line parsing.  Below are additional stream options that can be specified when running each example:

### Input Options

```
    input_URI            resource URI of the input stream (see table above)
  --input-width=WIDTH    explicitly request a resolution of the input stream
  --input-height=HEIGHT  (resolution is optional, except required for RTP)
  --input-codec=CODEC    RTP requires the codec to be set, one of these:
                             * h264, h265
                             * vp8, vp9
                             * mpeg2, mpeg4
                             * mjpeg
  --input-flip=FLIP      flip method to apply to input (excludes V4L2):
                             * none (default)
                             * counterclockwise
                             * rotate-180
                             * clockwise
                             * horizontal
                             * vertical
                             * upper-right-diagonal
                             * upper-left-diagonal
  --input-loop=LOOP      for file-based inputs, the number of loops to run:
                             * -1 = loop forever
                             *  0 = don't loop (default)
                             * >0 = set number of loops
```

### Output Options

```
    output_URI           resource URI of the output stream (see table above)
  --output-codec=CODEC   desired codec for compressed output streams:
                            * h264 (default), h265
                            * vp8, vp9
                            * mpeg2, mpeg4
                            * mjpeg
  --bitrate=BITRATE      desired target VBR bitrate for compressed streams,
                         in bits per second. The default is 4000000 (4 Mbps)
  --headless             don't create a default OpenGL GUI window
```

## Streaming Examples

Below are example commands of launching the `video-viewer` tool on various types of streams.  You can substitute the other programs for `video-viewer` in these commands, since they parse the same arguments.  In the [Source Code](#source-code) section of this page, you can browse the contents of the `video-viewer` source code to show how to use the `videoSource` and `videoOutput` APIs in your own applications.

### MIPI CSI cameras

MIPI CSI cameras are compact sensors that are acquired directly by the Jetson's hardware CSI/ISP interface.  Supported CSI cameras include:

* [Raspberry Pi Camera Module v2](https://www.raspberrypi.org/products/camera-module-v2/) (IMX219) for Jetson Nano and Jetson Xavier NX
* OV5693 sensor that comes with the Jetson TX1/TX2 devkits.  
* See the [Jetson Partner Supported Cameras](https://developer.nvidia.com/embedded/jetson-partner-supported-cameras) page for more sensors supported by the ecosystem.

Here's a few examples of launching with a MIPI CSI camera.  If you have multiple CSI cameras attached, subsitute the camera number for `0`:

```bash
$ video-viewer csi://0                        # MIPI CSI camera 0 (substitue other camera numbers)
$ video-viewer csi://0 output.mp4             # save output stream to MP4 file (H.264 by default)
$ video-viewer csi://0 rtp://<remote-ip>:1234 # broadcast output stream over RTP to <remote-ip>
```

By default, CSI cameras will be created with a 1280x720 resolution.  To specify a different resolution, use the `--input-width` and `input-height` options.  Note that the specified resolution must be one of the formats supported by the camera.

```bash
$ video-viewer --input-width=1920 --input-height=1080 csi://0
```

### V4L2 cameras

USB webcams are most commonly supported as V4L2 devices, for example Logitech [C270](https://www.logitech.com/en-us/product/hd-webcam-c270) or [C920](https://www.logitech.com/en-us/product/hd-pro-webcam-c920).

```bash
$ video-viewer v4l2:///dev/video0                 # /dev/video0 can be replaced with /dev/video1, ect.
$ video-viewer /dev/video0                        # dropping the v4l2:// protocol prefix is fine
$ video-viewer /dev/video0 output.mp4             # save output stream to MP4 file (H.264 by default)
$ video-viewer /dev/video0 rtp://<remote-ip>:1234 # broadcast output stream over RTP to <remote-ip>
```

> **note:**  if you have a MIPI CSI camera plugged in, it will also show up as `/dev/video0`.  Then if you plug in a USB webcam, that would show up as `/dev/video1`, so you would want to substitue `/dev/video1` in the commands above.  Using CSI cameras through V4L2 is unsupported in this project, as in V4L2 they use raw Bayer without ISP (instead, use CSI cameras as shown [above](#mipi-csi-cameras).

#### V4L2 Formats

When you run one of the jetson-inference programs on a V4L2 source, the different formats that the V4L2 camera supports will be logged to the terminal.  However you can also list these supported formats with the `v4l2-ctl` command:

```bash
$ sudo apt-get install v4l-utils
$ v4l2-ctl --device=/dev/video0 --list-formats-ext
```
 
By default, V4L2 cameras will be created using the camera format with the highest framerate that most closely matches the desired resolution (by default, that resolution is 1280x720).  The format with the highest framerate may be encoded (for example with H.264 or MJPEG), as USB cameras typically transmit uncompressed YUV/RGB at lower framerates.  In this case, the codec will be detected and the camera stream will automatically be decoded using the Jetson's hardware decoder to attain the highest framerate.

If you explicitly want to set the format used by the V4L2 camera, you can do so with the `--input-width`, `--input-height`, and `--input-codec` options.  Possible decoder codec options are `--input-codec=h264, h265, vp8, vp9, mpeg2, mpeg4, mjpeg`

```bash
$ video-viewer --input-width=1920 --input-height=1080 --input-codec=h264 /dev/video0
```
 
### RTP

RTP streams are broadcast to a particular host or multicast group over UDP/IP.  When recieving an RTP stream, additional options must be set (such as the resolution and codec) that match the source, because the RTP protocol doesn't include the ability to dynamically query these properties.

```bash
$ video-viewer --input-width=1280 --input-height=720 --input-codec=h264 rtp://@:1234         # recieve on localhost port 1234
$ video-viewer --input-width=1280 --input-height=720 --input-codec=h264 rtp://224.0.0.0:1234 # subscribe to multicast group
```

#### RTP Outputs

To transmit an RTP output stream, you needn't set the options above, but you can specify a different `--output-codec` if desired (the default is `--output-codec=h264`).  Possible encoder codec options are `--output-codec=h264, h265, vp8, vp9, mjpeg`.  You can also set the bitrate with `--bitrate` (the default is `--bitrate=4000000` or 4Mbps).

```bash
$ video-viewer --output-codec=h265 my_video.mp4 rtp://<remote-ip>:1234  # transmit a video file over RTP, encoded as H.265
$ video-viewer --bitrate=1000000 csi://0 rtp://<remote-ip>:1234         # transmit camera over RTP, encoded as H.264 @ 1Mbps 
```

When outputting RTP, you do need to explicitly set the IP address or hostname of the remote host or multicast group that the stream is being sent to (shown below as `<remote-ip>`).

### RTSP

RTSP streams are subscribed to from a remote host.  Unlike RTP, RTSP can dynamically query the stream properties (like resolution and codec), so these options needn't be explicitly provided.

```bash
$ video-viewer rtsp://<remote-ip>:1234 my_video.mp4	# subscribe to RTSP feed from <remote-ip>, port 1234 (and save it to file)
```

> **note:** RTSP is supported as an input only.  Outputting RTSP would require additional support for an RTSP server in GStreamer.

### Video Files

You can playback and record video files in MP4, MKV, AVI, and FLV formats.

```bash
$ video-viewer my_video.mp4						# display the video file
$ video-viewer my_video.mp4 rtp://<remote-ip>:1234	# transmit the video over RTP
$ video-viewer csi://0 my_video.mp4                    # record CSI camera to video file
$ video-viewer /dev/video0 my_video.mp4                # record V4L2 camera to video file
```

#### Codecs

The following codecs are supported:

* Decode:  H.264, H.265, VP8, VP9, MPEG-2, MPEG-4, MJPEG
* Encode:  H.264, H.265, VP8, VP9, MJPEG

When loading video files, the codec and resolution is automatically detected, so these needn't be set.
When saving video files, the default codec is H.264, but this can be set with the `--output-codec` option.

```bash
$ video-viewer --output-codec=h265 input.mp4 output.mp4  # transcode video to H.265
```

#### Resizing Inputs

When loading video files, the resolution is automatically detected.  However, if you would like the input video to be re-scaled to a different resolution, you can specify the `--input-width` and `--input-height` options:

```bash
$ video-viewer --input-width=640 --input-height=480 my_video.mp4	 # resize video to 640x480
```

#### Looping Inputs

By default, the stream will terminate once the end of the video is reached (EOS).  However, by specifying the `--loop` option, you can set the number of loops that you want the video to run for.  Possible options for `--loop` are:

* `-1` = loop forever
* ` 0` = don't loop (default)
* `>0` = set number of loops

```bash
$ video-viewer --loop=10 my_video.mp4    # loop the video 10 times
$ video-viewer --loop=-1 my_video.mp4    # loop the video forever (until user quits)
```

### Image Files

You can load/save image files in the following formats:

* Load:  JPG, PNG, TGA, BMP, GIF, PSD, HDR, PIC, and PNM (PPM/PGM binary)
* Save:  JPG, PNG, TGA, BMP

```bash
$ video-viewer input.jpg output.jpg	# load/save an image
```

You can loop images and image sequences, see the [Looping Inputs](#looping-inputs) section above.

#### Sequences

If the path is a directory or contains wildcards, all of the images will be loaded or saved sequentially (in alphanumeric order).

```bash
$ video-viewer input_dir/ output_dir/   # load all images from input_dir and save them to output_dir
$ video-viewer "*.jpg" output_%i.jpg    # load all jpg images and save them to output_0.jpg, output_1.jpg, ect
```

> *note:* when using wildcards, always enclose it in quotes (`".jpg"`). Otherwise, the OS will auto-expand the sequence and modify the order of arguments on the command-line, which can result in one of the input images being overwritten by the output.

When saving a sequence of images, if the path is just to a directory (`output_dir`), then the images will automatically be saved as JPG with the image number as it's filename (i.e. `output_dir/0.jpg`, `output_dir/1.jpg`, ect).  If you wish to specify the format, do so by using the printf-style `%i` in the path (i.e. `output_dir/image_%i.png`).  You can apply additional printf modifiers such as `%04i` to create filenames like `output_dir/0001.jpg`.


## Source Code

Streams are accessed using the [`videoSource`](https://github.com/dusty-nv/jetson-utils/video/videoSource.h) and [`videoOutput`](https://github.com/dusty-nv/jetson-utils/video/videoOutput.h) objects.  These have the ability to handle each of the above through a unified set of APIs.  The streams are identified via a resource URI.  The accepted formats and protocols of the resource URIs are documented below, along with example commands of using the `video-viewer` tool with them.  Note that you can substitute other examples such as `imagenet`, `detectnet`, `segnet` (and their respective `.py` Python versions) for `video-viewer` below, because they all accept the same command-line arguments.

##
<p align="right">Next | <b><a href="aux-image-manipulation.md">Image Manipulation with CUDA</a></b>
<p align="center"><sup>© 2016-2020 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>

