<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">

# Camera Streaming and Multimedia

jetson-inference supports streaming video feeds and images via a variety of interfaces and protocols, including MIPI CSI cameras, V4L2 cameras, RTP/RTSP over UDP/IP, video/image files, and sequences of images.  Streams are identified via a resource URI and accessed through the [`videoSource`](#videoSource) and [`videoOutput`](#videoOutput) APIs.

**Inputs**

|                  | Protocol     | URI Format                | Notes                                                    |
|------------------|--------------|---------------------------|----------------------------------------------------------|
| MIPI CSI cameras | `csi://`     | `csi://0`                 | CSI camera port 0                                        |
| V4L2 cameras     | `v4l2://`    | `v4l2:///dev/video0`      | V4L2 device 0                                            |
| RTP stream       | `rtp://`     | `rtp://@:1234`            | localhost, port 1234 (requires additional configuration) |
| RTSP stream      | `rtsp://`    | `rtsp://<remote-ip>:1234` | Replace `<remote-ip>` with remote host's IP or hostname  |
| Video file       | `file://`    | `file://my_video.mp4`     | Supports loading MP4, MKV, AVI, FLV (see codecs below)   |
| Image file       | `file://`    | `file://my_image.jpg`     | Supports loading JPG, PNG, TGA, BMP, GIF, ect.           |
| Image sequence   | `file://`    | `file://my_directory/`    | Searches for images in alphanumeric order                |

* Supported decoder codecs:  H.264, H.265, VP8, VP9, MPEG-2, MPEG-4, MJPEG

**Outputs**

|                  | Protocol     | URI Format                | Notes                                                    |
|------------------|--------------|---------------------------|----------------------------------------------------------|
| OpenGL display   | `display://` | `display://0`             | Creates GUI window on screen 0                           |
| RTP              | `rtp://`     | `rtp://<remote-ip>:1234`  | Replace `<remote-ip>` with remote host's IP or hostname  |
| Video file       | `file://`    | `file://my_video.mp4`     | Supports saving MP4, MKV, AVI, FLV (see codecs below)    |
| Image file       | `file://`    | `file://my_image.jpg`     | Supports saving JPG, PNG, TGA, BMP                       |
| Image sequence   | `file://`    | `file://image_%i.jpg`     | `%i` is replaced by the image number in the sequence     |

* Supported encoder codecs:  H.264, H.265, VP8, VP9, MJPEG



Streams are accessed using the [`videoSource`](https://github.com/dusty-nv/jetson-utils/video/videoSource.h) and [`videoOutput`](https://github.com/dusty-nv/jetson-utils/video/videoOutput.h) objects.  These have the ability to handle each of the above through a unified set of APIs.  The streams are identified via a resource URI.  The accepted formats and protocols of the resource URIs are documented below, along with example commands of using the `video-viewer` tool with them.  Note that you can substitute other examples such as `imagenet`, `detectnet`, `segnet` (and their respective `.py` Python versions) for `video-viewer` below, because they all accept the same command-line arguments.

