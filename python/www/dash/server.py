#!/usr/bin/env python3
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
import threading
import multiprocessing
import ssl
import time

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.client import ServerProxy

from jetson_utils import videoSource, videoOutput, Log


# TODO pass videoSource/videoOutput flags  (ssl_certs, ect)
class Stream:
    def __init__(self, server, name, source):
        if not name.startswith('/'):   # make sure all routes start with '/'
            name = '/' + name
            
        # enable HTTPS/SSL
        video_args = None
        
        if server.ssl_context is not None:
            video_args = [f"--ssl-cert={server.ssl_context[0]}", f"--ssl-key={server.ssl_context[1]}"]
            
        self.server = server
        self.name = name
        self.source = videoSource(source, argv=video_args)
        self.output = videoOutput(f"webrtc://@:{self.server.webrtc_port}{self.name}", argv=video_args)
        self.frame_count = 0
        
    def process(self):
        try:
            img = self.source.Capture()
        except Exception as error:
            # TODO check if stream is still open, if not reconnect?
            Log.Error(f"{error}")
            return
            
        if self.frame_count % 25 == 0 or self.frame_count < 15:
            Log.Verbose(f"[{self.server.name}] {self.name} -- captured frame {self.frame_count}  ({img.width}x{img.height})")

        self.output.Render(img)
        self.frame_count += 1
       
    def get_config(self):
        # TODO add stats or runtime_stats option for easy frontend state-change comparison?
        # the videoOptions could be dynamic as well... (i.e. framerate - actually that is not?)
        return {
            "name" : self.name,
            "source" : self.source.GetOptions(),
            "output" : self.output.GetOptions(),
            #'frame_count' : self.frame_count 
        }
        
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

    def get_config(self):
        return {stream.name : stream.get_config() for stream in self.streams}

        
class Server:
    """
    Backend inference media streaming server for handling streams, models, datasets, ect.
    It typically runs in it's own process and uses RPC for communication. 
    """
    def __init__(self, name="inference-backend", host="0.0.0.0", rpc_port=49565, webrtc_port=49567, ssl_context=None):
        self.name = name
        self.host = host
        self.rpc_port = rpc_port
        self.webrtc_port = webrtc_port
        self.ssl_context = ssl_context
        self.streams = Streams(self)
        self.run_flag = False  # this gets set to true when initialized successfully
        
    def init(self):
        """
        This gets called once the beginning of run() from within the process.
        """
        self.rpc_server = SimpleXMLRPCServer((self.host, self.rpc_port), allow_none=True)  # logRequests = False 
        
        #if self.ssl_context is not None:
        #    self.rpc_server.socket = ssl.wrap_socket(self.rpc_server.socket, certfile=self.ssl_context[0], keyfile=self.ssl_context[1], 
        #                                             server_side=True, ssl_version=ssl.PROTOCOL_TLS)
            
        self.rpc_server.register_instance(self, allow_dotted_names=True)
        
        # run the RPC server in it's own thread
        self.rpc_thread = threading.Thread(target=lambda: self.rpc_server.serve_forever(), name=f"{self.name}-rpc")
        self.rpc_thread.start()     
        
        Log.Info(f"[{self.name}] RPC server is running @ http://{self.host}:{self.rpc_port}")

        # indicate that server is ready to run
        self.run_flag = True

    def start(self):
        """
        Launch the server running in a new process.
        Returns the RPC proxy object for clients to call.
        """
        # we don't need the dash/webserver stuff, so use spawn instead of fork
        # TODO look into the memory savings/implications of this
        # https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
        multiprocessing.set_start_method("spawn")  
        
        # start the server process
        self.os_process = multiprocessing.Process(target=self.run, name=self.name, daemon=True)  # use daemon=True so process automatically exits when parent process exits
        self.os_process.start()
            
        # create the RPC proxy object for the caller
        self.rpc_proxy = ServerProxy(f"http://{self.host}:{self.rpc_port}")
        
        # rudimentary check to confirm the process started/initialized ok
        # TODO put this in a loop and re-try multiple times for when we start loading models/ect at server start-up
        max_retries = 10
        retry_count = 0
        
        for retry_count in range(max_retries):
            time.sleep(0.5)
        
            try:
                if self.rpc_proxy.is_running():
                    break
                else:
                    if retry_count == max_retries - 1:
                        raise Exception(f"[{self.name}] failed to start running")
            except ConnectionRefusedError as error:
                Log.Verbose(f"[{self.name}] {error}")
            
            #Log.Verbose(f"[{self.name}] waiting for connection... (retry {retry_count+1} of {max_retries})")
            
        return self.rpc_proxy
        
    def stop(self):
        """
        Signal the process to stop running.
        """
        print(f"[{self.name}] stopping...")
        self.run_flag = False
        self.rpc_server._BaseServer__shutdown_request = True 
        self.rpc_server.server_close()
        #self.process.join()
        
    def run(self):
        """
        Run forever - this automatically gets called by the process when it starts.
        """
        self.init()

        while self.run_flag:
            self.process()
          
        print(f"[{self.name}] stopped")
        
    def is_running(self):
        """
        Returns true if the process is initialized and running.
        """
        return self.run_flag
        
    def process(self):
        """
        Perform one interation of the processing loop.
        """
        for i in range(len(self.streams.streams)):     # TODO don't spin if no streams
            self.streams.streams[i].process()

    def get_config(self):
        """
        Return a dict of the server's streams/models/datasets config
        """
        return {
            "streams" : self.streams.get_config(),
            "models" : [],  # TODO implement models/streams
            "datasets" : []
        }