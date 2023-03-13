#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

from jetson_utils import videoSource, videoOutput, Log

import pprint
import traceback

class Stream:
    """
    Represents a pipeline from a video source -> processing -> video output
    """
    def __init__(self, server, name, source, models=[]):
        # make sure all routes start with '/'
        if not name.startswith('/'):   
            name = '/' + name
            
        # enable HTTPS/SSL
        video_args = None
        
        if server.ssl_cert and server.ssl_key:
            video_args = [f"--ssl-cert={server.ssl_cert}", f"--ssl-key={server.ssl_key}"]
            
        video_args += ['--input-codec=mjpeg', '--output-encoder=cpu']
        
        self.server = server
        self.name = name
        self.frame_count = 0
        
        # create video interfaces
        self.source = videoSource(source, argv=video_args)
        self.output = videoOutput(f"webrtc://@:{self.server.webrtc_port}{self.name}", argv=video_args)
        
        # lookup models
        self.models = []
        
        if models is None:
            models = []
            
        if isinstance(models, str):
            models = [models]

        for model in models:
            if model in server.resources['models']:
                self.models.append(server.resources['models'][model].clone(stream=self))
            else:
                Log.Verbose(f"[{self.server.name}] model '{model}' was not loaded on server")

    def process(self):
        """
        Perform one capture/process/output iteration
        """
        try:
            img = self.source.Capture()
            
            if img is None:  # timeout
                return
                
            for model in self.models:
                model.process(img)
                
            for model in self.models:
                model.visualize(img)
        except:
            # TODO check if stream is still open, if not reconnect?
            traceback.print_exc()
            return
            
        if self.frame_count % 25 == 0 or self.frame_count < 15:
            Log.Verbose(f"[{self.server.name}] {self.name} -- captured frame {self.frame_count}  ({img.width}x{img.height})")

        self.output.Render(img)
        self.frame_count += 1
       
    def get_config(self):
        """
        TODO add stats or runtime_stats option for easy frontend state-change comparison?
        the videoOptions could be dynamic as well... (i.e. framerate - actually that is not?)
        """
        return {
            "name" : self.name,
            "source" : self.source.GetOptions(),
            "output" : self.output.GetOptions(),
            "models" : [model.name for model in self.models]
            #'frame_count' : self.frame_count 
        }

"""       
class Streams:
    def __init__(self, server):
        self.server = server
        self.streams = []
        
    def add(self, name, source):
        print(f"[{self.server.name}] adding stream {name}  source {source}")
        stream = Stream(self.server, name, source)
        self.streams.append(stream)
        return stream.get_config()  # return the config dict, as this is called over RPC 
        
    def keys(self):
        return [stream.name for stream in self.streams]
        
    def __len__(self):
        return len(self.streams)
        
    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self.streams):
                raise IndexError("stream index was out of range")
            return self.streams[key]
        elif isinstance(key, str):
            for stream in self.streams:
                if stream.name == key:
                    return stream
            raise KeyError(f"couldn't find stream '{key}'")
        else:
            raise TypeError("index/key must be of type int or string")

    def get_config(self, key):
        return self[key].get_config()
        
    def list_streams(self):
        return {stream.name : stream.get_config() for stream in self.streams}
"""