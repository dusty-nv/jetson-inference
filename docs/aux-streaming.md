<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="../README.md#appendix">Back</a> | <a href="aux-image.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Appendix</sup></p>  

# Camera Streaming and Multimedia

This project supports capturing and streaming video feeds & static images via a variety of interfaces and protocols, including:

* [MIPI CSI cameras](#mipi-csi-cameras)
* [V4L2 cameras](#v4l2-cameras)
* [WebRTC](#webrtc)
* [RTP](#rtp) / [RTSP](#rtsp) 
* [Videos](#video-files) & [Images](#image-files)
* [Image sequences](#image-files)
* [OpenGL windows](#output-streams)

Streams are identified via a resource URI and are acquired through the [`videoSource`](#source-code) and [`videoOutput`](#source-code) APIs.  These settings can be configured via command-line arguments or in your application's [source code](#source-code).  The tables below show the supported input/output protocols and URI formats:

### Input Streams

|                                      | Protocol     | Resource URI                | Notes                                                    |
|--------------------------------------|--------------|-----------------------------|----------------------------------------------------------|
| [MIPI CSI camera](#mipi-csi-cameras) | `csi://`     | `csi://0`                   | CSI camera 0 (substitute other camera numbers for `0`)   |
| [V4L2 camera](#v4l2-cameras)         | `v4l2://`    | `v4l2:///dev/video0`        | V4L2 device 0 (substitute other camera numbers for `0`)  |
| [WebRTC stream](#webrtc)             | `webrtc://`  | `webrtc://@:8554/my_input`  | From browser webcam to localhost, port 8554 (requires HTTPS/SSL) |
| [RTP stream](#rtp)                   | `rtp://`     | `rtp://@:1234`              | localhost, port 1234 (requires additional configuration) |
| [RTSP stream](#rtsp)                 | `rtsp://`    | `rtsp://<remote-ip>:1234`   | Replace `<remote-ip>` with remote host's IP or hostname  |
| [Video file](#video-files)           | `file://`    | `file://my_video.mp4`       | Supports loading MP4, MKV, AVI, FLV (see codecs below)   |
| [Image file](#image-files)           | `file://`    | `file://my_image.jpg`       | Supports loading JPG, PNG, TGA, BMP, GIF, ect.           |
| [Image sequence](#image-files)       | `file://`    | `file://my_directory/`      | Searches for images in alphanumeric order                |

* Supported decoder codecs:  H.264, H.265, VP8, VP9, MPEG-2, MPEG-4, MJPEG
* The `file://`, `v4l2://`, and `csi://` protocol prefixes can be omitted from the URI as shorthand

### Output Streams

|                                  | Protocol     | Resource URI                | Notes                                                    |
|----------------------------------|--------------|-----------------------------|----------------------------------------------------------|
| [WebRTC stream](#webrtc)         | `webrtc://`  | `webrtc://@:8554/my_output` | Send to browser, port 8554, stream name `"my_output"`    |
| [RTP stream](#rtp)               | `rtp://`     | `rtp://<remote-ip>:1234`    | Replace `<remote-ip>` with remote host's IP or hostname  |
| [RTSP stream](#rtsp)             | `rtsp://`    | `rtsp://@:1234/my_output`   | Reachable at `rtsp://<jetson-ip>:1234/my_output`         |
| [Video file](#video-files)       | `file://`    | `file://my_video.mp4`       | Supports saving MP4, MKV, AVI, FLV (see codecs below)    |
| [Image file](#image-files)       | `file://`    | `file://my_image.jpg`       | Supports saving JPG, PNG, TGA, BMP                       |
| [Image sequence](#image-files)   | `file://`    | `file://image_%i.jpg`       | `%i` is replaced by the image number in the sequence     |
| [OpenGL window](#output-streams) | `display://` | `display://0`               | Creates GUI window on screen 0                           |

* Supported encoder codecs:  H.264, H.265, VP8, VP9, MJPEG
* The `file://` protocol prefixes can be omitted from the URI as shorthand
* By default, an OpenGL display window will be created unless `--headless` is specified

## Command-Line Arguments

Each example C++ and Python program from jetson-inference accepts the same set of command-line arguments for specifying stream URIs and additional options. So these options can be used on any of the examples (e.g. [`imagenet`](../examples/imagenet/imagenet.cpp)/[`imagenet.py`](../examples/python/imagenet.py), [`detectnet`](../examples/detectnet/detectnet.cpp)/[`detectnet.py`](../examples/python/detectnet.py), [`segnet`](../examples/segnet/segnet.cpp)/[`segnet.py`](../examples/python/segnet.py), [`video-viewer`](https://github.com/dusty-nv/jetson-utils/tree/master/video/video-viewer/video-viewer.cpp)/[`video-viewer.py`](https://github.com/dusty-nv/jetson-utils/tree/master/python/examples/video-viewer.py), ect).  These command-line arguments generally take the form:

```bash
$ imagenet [options] input_URI [output_URI]  # output URI is optional (default is display://0)
```

where the input and output URIs are specified by two positional arguments.  For example:

```bash
$ imagenet input.jpg output.jpg              # classify input.jpg, save as output.jpg
```

As mentioned above, any of the examples from jetson-inference can be substituted here, since they use the same command-line parsing.  Below are additional stream options that can be specified when running each program:

#### Input Options

```
    input                resource URI of the input stream, for example:
                             * /dev/video0               (V4L2 camera #0)
                             * csi://0                   (MIPI CSI camera #0)
                             * rtp://@:1234              (RTP stream)
                             * rtsp://user:pass@ip:1234  (RTSP stream)
                             * webrtc://@:1234/my_stream (WebRTC stream)
                             * file://my_image.jpg       (image file)
                             * file://my_video.mp4       (video file)
                             * file://my_directory/      (directory of images)
  --input-width=WIDTH    explicitly request a width of the stream (optional)
  --input-height=HEIGHT  explicitly request a height of the stream (optional)
  --input-rate=RATE      explicitly request a framerate of the stream (optional)
  --input-save=FILE      path to video file for saving the input stream to disk
  --input-codec=CODEC    RTP requires the codec to be set, one of these:
                             * h264, h265
                             * vp8, vp9
                             * mpeg2, mpeg4
                             * mjpeg
  --input-decoder=TYPE   the decoder engine to use, one of these:
                             * cpu
                             * omx  (aarch64/JetPack4 only)
                             * v4l2 (aarch64/JetPack5 only)
  --input-flip=FLIP      flip method to apply to input:
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

#### Output Options

```
    output               resource URI of the output stream, for example:
                             * file://my_image.jpg       (image file)
                             * file://my_video.mp4       (video file)
                             * file://my_directory/      (directory of images)
                             * rtp://<remote-ip>:1234    (RTP stream)
                             * rtsp://@:8554/my_stream   (RTSP stream)
                             * webrtc://@:1234/my_stream (WebRTC stream)
                             * display://0               (OpenGL window)
  --output-codec=CODEC   desired codec for compressed output streams:
                            * h264 (default), h265
                            * vp8, vp9
                            * mpeg2, mpeg4
                            * mjpeg
  --output-encoder=TYPE  the encoder engine to use, one of these:
                            * cpu
                            * omx  (aarch64/JetPack4 only)
                            * v4l2 (aarch64/JetPack5 only)
  --output-save=FILE     path to a video file for saving the compressed stream
                         to disk, in addition to the primary output above
  --bitrate=BITRATE      desired target VBR bitrate for compressed streams,
                         in bits per second. The default is 4000000 (4 Mbps)
  --headless             don't create a default OpenGL GUI window
```

Below are example commands of launching the `video-viewer` tool on various types of streams.  You can substitute the other programs for `video-viewer` in these commands, since they parse the same arguments.  In the [Source Code](#source-code) section of this page, you can browse the contents of the `video-viewer` source code to show how to use the `videoSource` and `videoOutput` APIs in your own applications.

## MIPI CSI cameras

MIPI CSI cameras are compact sensors that are acquired directly by the Jetson's hardware CSI/ISP interface.  Supported CSI cameras include:

* [Raspberry Pi Camera Module v2](https://www.raspberrypi.org/products/camera-module-v2/) (IMX219) for Jetson Nano and Jetson Xavier NX
* OV5693 camera module from the Jetson TX1/TX2 devkits.  
* See the [Jetson Partner Supported Cameras](https://developer.nvidia.com/embedded/jetson-partner-supported-cameras) page for more sensors supported by the ecosystem.

Here's a few examples of launching with a MIPI CSI camera.  If you have multiple CSI cameras attached, subsitute the camera number for 0:

```bash
$ video-viewer csi://0                        # MIPI CSI camera 0 (substitue other camera numbers)
$ video-viewer csi://0 output.mp4             # save output stream to MP4 file (H.264 by default)
$ video-viewer csi://0 rtp://<remote-ip>:1234 # broadcast output stream over RTP to <remote-ip>
```

By default, CSI cameras will be created with a 1280x720 resolution.  To specify a different resolution, use the `--input-width` and `input-height` options.  Note that the specified resolution must match one of the formats supported by the camera.

```bash
$ video-viewer --input-width=1920 --input-height=1080 csi://0
```

## V4L2 cameras

USB webcams are most commonly supported as V4L2 devices, for example Logitech [C270](https://www.logitech.com/en-us/product/hd-webcam-c270) or [C920](https://www.logitech.com/en-us/product/hd-pro-webcam-c920).

```bash
$ video-viewer v4l2:///dev/video0                 # /dev/video0 can be replaced with /dev/video1, ect.
$ video-viewer /dev/video0                        # dropping the v4l2:// protocol prefix is fine
$ video-viewer /dev/video0 output.mp4             # save output stream to MP4 file (H.264 by default)
$ video-viewer /dev/video0 rtp://<remote-ip>:1234 # broadcast output stream over RTP to <remote-ip>
```

> **note:**  if you have a MIPI CSI camera plugged in, it will also show up as `/dev/video0`.  Then if you plug in a USB webcam, that would show up as `/dev/video1`, so you would want to substitue `/dev/video1` in the commands above.  Using CSI cameras through V4L2 is unsupported in this project, because through V4L2 they use raw Bayer without ISP (instead, use CSI cameras as shown [above](#mipi-csi-cameras)).

#### V4L2 Formats

By default, V4L2 cameras will be created using the camera format with the highest framerate that most closely matches the desired resolution (by default, that resolution is 1280x720).  The format with the highest framerate may be encoded (for example with H.264 or MJPEG), as USB cameras typically transmit uncompressed YUV/RGB at lower framerates.  In this case, that codec will be detected and the camera stream will automatically be decoded using the Jetson's hardware decoder to attain the highest framerate.

If you explicitly want to choose the format used by the V4L2 camera, you can do so with the `--input-width`, `--input-height`, and `--input-codec` options.  Possible decoder codec options are `--input-codec=h264, h265, vp8, vp9, mpeg2, mpeg4, mjpeg`

```bash
$ video-viewer --input-width=1920 --input-height=1080 --input-codec=h264 /dev/video0
```

When you run one of the jetson-inference programs on a V4L2 source, the different formats that the V4L2 camera supports will be logged to the terminal.  However you can also list these supported formats with the `v4l2-ctl` command:

```bash
$ sudo apt-get install v4l-utils
$ v4l2-ctl --device=/dev/video0 --list-formats-ext
```
  
## WebRTC

This projects includes a built-in WebRTC server (input/output) for streaming video to/from client web browsers.  You can use this for conveniently viewing video streams when your Jetson is headless and doesn't have a display attached, or for easily building interactive webapps that use Jetson and edge AI on the backend.  Tested browsers include Chrome/Chromium, mobile Android, and mobile iOS (Safari). 

```bash
$ video-viewer /dev/video0 webrtc://@:8554/my_output               # send V4L2 webcam to browser
$ video-viewer webrtc://@:8554/my_input output.mp4                 # receive browser webcam (requires HTTPS/SSL) and save to MP4
$ video-viewer webrtc://@:8554/my_input webrtc://@:8554/my_output  # receieve + send (full-duplex loopback)
```

> **note**: receiving browser webcams requires [HTTPS/SSL](webrtc-server.md#enabling-https--ssl) to be enabled

You should then be able to navigate your browser to `https://<JETSON-IP>:8554` to view the stream.  There's an entire section of the Hello AI World tutorial dedicated to using WebRTC and building applications with various webapp frameworks:

* [WebRTC Server](webrtc-server.md)
* [WebAPP Frameworks](../README.md#webapp-frameworks)

## RTP

RTP network streams are broadcast to a particular host or multicast group over UDP/IP.  When recieving an RTP stream, the codec must be specified (`--input-codec`), because RTP doesn't have the ability to dynamically query this.  This will use RTP as input from another device:

```bash
$ video-viewer --input-codec=h264 rtp://@:1234         # recieve on localhost port 1234
$ video-viewer --input-codec=h264 rtp://224.0.0.0:1234 # subscribe to multicast group
```

The commands above specify RTP as the input source, where another remote host on the network is streaming to the Jetson.  However, you can also output an RTP stream from your Jetson and transmit it to another remote host on the network.

#### Transmitting RTP

To transmit an RTP output stream, specify the target IP/port as the `output_URI`. If desired, you can specify the bitrate (the default is `--bitrate=4000000` or 4Mbps) and/or the output codec (the default is `--output-codec=h264`) which can be `h264, h265, vp8, vp9, mjpeg`

```bash
$ video-viewer --bitrate=1000000 csi://0 rtp://<remote-ip>:1234         # transmit camera over RTP, encoded as H.264 @ 1Mbps 
$ video-viewer --output-codec=h265 my_video.mp4 rtp://<remote-ip>:1234  # transmit a video file over RTP, encoded as H.265
```

When outputting RTP, you need to explicitly set the IP address or hostname of the remote host (or multicast group) that the stream is being sent to (shown above as `<remote-ip>`).  See below for some pointers on viewing the RTP stream from a PC.

#### Viewing RTP Remotely

If your Jetson is transmitting RTP to another remote host (like a PC), here are some example commands that you can use to view the stream:

* Using GStreamer:
	* [Install GStreamer](https://gstreamer.freedesktop.org/documentation/installing/index.html) and run this pipeline (replace `port=1234` with the port you are using)
	
	```bash
	$ gst-launch-1.0 -v udpsrc port=1234 \
	caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! \
	rtph264depay ! decodebin ! videoconvert ! autovideosink
	```
	
* Using VLC Player:
	* Create a SDP file (.sdp) with the following contents (replace `1234` with the port you are using)
	
	```
     c=IN IP4 127.0.0.1
     m=video 1234 RTP/AVP 96
     a=rtpmap:96 H264/90000
	```
	
	* Open the stream in VLC by double-clicking the SDP file
	* You may want to reduce the `File caching` and `Network caching` settings in VLC as [shown here](https://www.howtogeek.com/howto/windows/fix-for-vlc-skipping-and-lagging-playing-high-def-video-files/)
	
* If your remote host is another Jetson:
	* Use the same `video-viewer` command as [above](#rtp) (replace `1234` with the port you are using)
	
	```bash
	$ video-viewer --input-codec=h264 rtp://@:1234
     ```

## RTSP

RTSP network streams are subscribed to from a remote host over UDP/IP.  Unlike RTP, RTSP can dynamically query the stream properties (like resolution and codec), so these options don't need to be explicitly provided.

#### RTSP Input 

To connect to an RTSP source, supply the IP address and URL of the RTSP server offering it:

```bash
$ video-viewer rtsp://<remote-ip>:1234 my_video.mp4      # subscribe to RTSP feed from <remote-ip>, port 1234 (and save it to file)
$ video-viewer rtsp://username:password@<remote-ip>:1234 # with authentication (replace username/password with credentials)
```

You might need to supply the username/password credentials in the URL if the RTSP server has authentication enabled.

#### RTSP Output

jetson-utils includes a built-in RTSP server using GStreamer for sending compressed RTSP streams to multiple clients:

```bash
$ video-viewer /dev/video0 rtsp://@:1234/my_output                 # stream a V4L2 camera out over RTSP 
$ video-viewer rtsp://<remote-ip>:1234/input rtsp://@:1234/output  # subscribe to an RTSP feed, and relay it (loopback)
```

> **note:** SSL encryption can be enabled for RTSP output in the [same way](webrtc-server.md#enabling-https--ssl) that it is for WebRTC

You should then be able to open and view the stream from an RTSP client (like VLC player) at the URL `rtsp://<jetson-ip>:1234/my_output`

## Video Files

You can playback and record compressed video files in MP4, MKV, AVI, and FLV formats (in addition to uncontainerized H264/H265).

```bash
# playback
$ video-viewer my_video.mp4                              # display the video file
$ video-viewer my_video.mp4 rtp://<remote-ip>:1234       # transmit the video over RTP

# recording
$ video-viewer csi://0 my_video.mp4                      # record CSI camera to video file
$ video-viewer /dev/video0 my_video.mp4                  # record V4L2 camera to video file
```

#### Codecs

When loading video files, the codec and resolution is automatically detected, so these don't need to be set.
When saving video files, the default codec is H.264, but this can be set with the `--output-codec` option.

```bash
$ video-viewer --output-codec=h265 input.mp4 output.mp4  # transcode video to H.265
```

The following codecs are supported:

* Decode - `h264, h265, vp8, vp9, mpeg2, mpeg4, mjpeg`
* Encode - `h264, h265, vp8, vp9, mjpeg`


#### Resizing Inputs

When loading video files, the resolution is automatically detected.  However, if you would like the input video to be re-scaled to a different resolution, you can specify the `--input-width` and `--input-height` options:

```bash
$ video-viewer --input-width=640 --input-height=480 my_video.mp4  # resize video to 640x480
```

#### Looping Inputs

By default, the video will terminate once the end of stream (EOS) is reached.  However, by specifying the `--loop` option, you can set the number of loops that you want the video to run for.  Possible options for `--loop` are:

* `-1` = loop forever
* &nbsp;` 0` = don't loop (default)
* `>0` = set number of loops

```bash
$ video-viewer --loop=10 my_video.mp4    # loop the video 10 times
$ video-viewer --loop=-1 my_video.mp4    # loop the video forever (until user quits)
```

#### Secondary Destination

Sometimes you may wish to save the unprocessed camera feed (or the post-processed video) to disk in addition to the primary output stream.  For incoming inputs that are already compressed (for example, an H264-encoded camera or network stream), the `--input-save=<FILE>` option can be used to dump the encoded video to disk before it's decoded and processed.  It supports MP4, MKV, AVI, FLV, and raw H264/H265.

For output streams that are to be compressed (i.e. network streams like WebRTC/RTP/RTSP) then the `--output-save=<FILE>` option will record the processed video to disk in addition to it's primary output.  To save an output video file while also displaying it on a screen attached to your Jetson (which doesn't undergo compression), just use the method above for [recording video](#video-files) and an OpenGL GUI window will automatically open.


``` bash
$ detectnet --input-codec=h264 --input-save=camera_dump.mp4 /dev/video0       # save incoming/unprocessed video
$ detectnet --output-save=post_dump.mp4 /dev/video0 rtsp://@:1234/my_stream   # save outgoing/processed video
```

> **note:** `--input-save` and `--output-save` can only be used in conjunction with streams that are already compressed/encoded.

The first command will dump the original camera video, and the second will dump it after processing (e.g. including bounding boxes, ect)

## Image Files

You can load/save image files in the following formats:

* Load:  JPG, PNG, TGA, BMP, GIF, PSD, HDR, PIC, and PNM (PPM/PGM binary)
* Save:  JPG, PNG, TGA, BMP

```bash
$ video-viewer input.jpg output.jpg	# load/save an image
```

You can also loop images and image sequences - see the [Looping Inputs](#looping-inputs) section above.

#### Sequences

If the path is a directory or contains wildcards, all of the images will be loaded/saved sequentially (in alphanumeric order).

```bash
$ video-viewer input_dir/ output_dir/   # load all images from input_dir and save them to output_dir
$ video-viewer "*.jpg" output_%i.jpg    # load all jpg images and save them to output_0.jpg, output_1.jpg, ect
```

> **note:** when using wildcards, always enclose it in quotes (`"*.jpg"`). Otherwise, the OS will auto-expand the sequence and modify the order of arguments on the command-line, which may result in one of the input images being overwritten by the output.

When saving a sequence of images, if the path is just to a directory (`output_dir`), then the images will automatically be saved as JPG with the format `output_dir/%i.jpg`, using the image number as it's filename (`output_dir/0.jpg`, `output_dir/1.jpg`, ect).  

If you wish to specify the filename format, do so by using the printf-style `%i` in the path (`output_dir/image_%i.png`).  You can apply additional printf modifiers such as `%04i` to create filenames like `output_dir/image_0001.jpg`.


## Source Code

Streams are accessed using the [`videoSource`](https://github.com/dusty-nv/jetson-utils/tree/master/video/videoSource.h) and [`videoOutput`](https://github.com/dusty-nv/jetson-utils/tree/master/video/videoOutput.h) objects.  These have the ability to handle each of the types of streams from above through a unified set of APIs.  Images can be captured and output in the following data formats:  

| Format string | [`imageFormat` enum](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/group__imageFormat.html#ga931c48e08f361637d093355d64583406) | Data Type | Bit Depth |
|---------------|------------------|-----------|-----------|
| `rgb8`        | `IMAGE_RGB8`     | `uchar3`  | 24        |
| `rgba8`       | `IMAGE_RGBA8`    | `uchar4`  | 32        |
| `rgb32f`      | `IMAGE_RGB32F`   | `float3`  | 96        |
| `rgba32f`     | `IMAGE_RGBA32F`  | `float4`  | 128       |

* the Data Type and [`imageFormat`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/group__imageFormat.html#ga931c48e08f361637d093355d64583406) enum are C++ types
* in Python, the format string can be passed to `videoSource.Capture()` to request a specific format (the default is `rgb8`)
* in C++, the `videoSource::Capture()` template will infer the format from the data type of the output pointer

To convert images to/from different formats, see the [Image Manipulation with CUDA](aux-image.md) page for more info.

Below is the source code to [`video-viewer.py`](https://github.com/dusty-nv/jetson-utils/tree/master/python/examples/video-viewer.py) and [`video-viewer`](https://github.com/dusty-nv/jetson-utils/tree/master/video/video-viewer/video-viewer.cpp), slightly abbreviated to improve readability:

### Python
```python
import sys
import argparse

from jetson_utils import videoSource, videoOutput

# parse command line
parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
args = parser.parse_known_args()[0]

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)    # default:  options={'width': 1280, 'height': 720, 'framerate': 30}
output = videoOutput(args.output, argv=sys.argv)  # default:  options={'codec': 'h264', 'bitrate': 4000000}

# capture frames until end-of-stream (or the user exits)
while True:
    # format can be:   rgb8, rgba8, rgb32f, rgba32f (rgb8 is the default)
    # timeout can be:  -1 for infinite timeout (blocking), 0 to return immediately, >0 in milliseconds (default is 1000ms)
    image = input.Capture(format='rgb8', timeout=1000)  
	
    if image is None:  # if a timeout occurred
        continue
		
    output.Render(image)

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
```

To hardcode video configuration settings in Python, you can pass an optional `options` dictionary to the videoSource/videoOutput initializer, which roughly corresponds to the [`videoOptions`](https://github.com/dusty-nv/jetson-utils/blob/master/video/videoOptions.h) structure in C++:

``` python
input = videoSource("/dev/video0", options={'width': 1280, 'height': 720, 'framerate': 30, 'flipMethod': 'rotate-180'})
output = videoOutput("my_video.mp4", options={'codec': 'h264', 'bitrate': 4000000})
```

The input settings will use the closest resolution/framerate available, but it's recommend to check your camera's [supported formats](#v4l2-formats) first.

### C++
```c++
#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>

int main( int argc, char** argv )
{
    // create input/output streams
    videoSource* input = videoSource::Create(argc, argv, ARG_POSITION(0));
    videoOutput* output = videoOutput::Create(argc, argv, ARG_POSITION(1));
	
    if( !input )
        return 0;

    // capture/display loop
    while( true )
    {
        uchar3* image = NULL;  // can be uchar3, uchar4, float3, float4
        int status = 0;        // see videoSource::Status (OK, TIMEOUT, EOS, ERROR)
		
        if( !input->Capture(&image, 1000, &status) )  // 1000ms timeout (default)
        {
            if( status == videoSource::TIMEOUT )
                continue;
				
            break; // EOS
        }

        if( output != NULL )
        {
            output->Render(image, inputStream->GetWidth(), inputStream->GetHeight());

            if( !output->IsStreaming() )  // check if the user quit
                break;
        }
    }

    // destroy resources
    SAFE_DELETE(input);
    SAFE_DELETE(output);
}
```

To create the interfaces programatically from coded settings, the [`videoOptions`](https://github.com/dusty-nv/jetson-utils/blob/master/video/videoOptions.h) struct can also be populated like this:

```c++
videoOptions options;

options.width = 1280;
options.height = 720;
options.frameRate = 30;
options.flipMethod = videoOptions::FLIP_ROTATE_180;

videoSource* input = videoSource::Create("/dev/video0", options);
```

The input settings will use the closest resolution/framerate available, but it's recommend to check your camera's [supported formats](#v4l2-formats) first.

<p align="right">Next | <b><a href="aux-image.md">Image Manipulation with CUDA</a></b>
<p align="center"><sup>© 2016-2020 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>

