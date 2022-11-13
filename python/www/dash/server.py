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

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.client import ServerProxy

from jetson_utils import videoSource, videoOutput

# TODO using logging APIs...
# TODO pass videoSource/videoOutput flags  (ssl_certs, ect)
class Stream:
    def __init__(self, server, name, source):
        if not name.startswith('/'):   # make sure all routes start with '/'
            name = '/' + name
            
        self.server = server
        self.name = name
        self.source_path = source
        self.source = videoSource(self.source_path)
        self.output_path = f'webrtc://@:{self.server.webrtc_port}{self.name}'
        self.output = videoOutput(self.output_path)
        self.frame_count = 0
        
    def process(self):
        img = self.source.Capture()
        
        if self.frame_count % 25 == 0 or self.frame_count < 15:
            print(f'[{self.server.name}] {self.name} -- captured frame {self.frame_count}  ({img.width}x{img.height})')

        self.output.Render(img)
        self.frame_count += 1
       
    @property
    def state(self):
        return {
            'name' : self.name,
            'source' : self.source_path,
            'output' : self.output_path,
            'frame_count' : self.frame_count 
        }
        
class Streams:
    def __init__(self, server):
        self.server = server
        self.streams = []
        
    def add(self, name, source):
        print(f'[{self.server.name}] adding stream {name}  source {source}')
        stream = Stream(self.server, name, source)
        self.streams.append(stream)
        return stream.state
        
    def __len__(self):
        return len(self.streams)
        
    def __getitem__(self, key):
        return self.streams[key]
        
class Server:
    """
    Media server for handling streams, models, datasets, ect.
    It typically runs in it's own process and uses RPC for communication. 
    """
    def __init__(self, name='webrtc-dash-server', host='0.0.0.0', rpc_port=49565, webrtc_port=49567, ssl_context=None):
        self.name = name
        self.host = host
        self.rpc_port = rpc_port
        self.webrtc_port = webrtc_port
        self.ssl_context = ssl_context
        self.run_flag = True
        self.streams = Streams(self)

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
        self.rpc_thread = threading.Thread(target=lambda: self.rpc_server.serve_forever(), name=f'{self.name}-rpc')
        self.rpc_thread.start()        
        
        print(f'[{self.name}] running RPC server @ http://{self.host}:{self.rpc_port}')
        
    def start(self):
        """
        Launch the server running in a new process.
        Returns the RPC proxy object for clients to call.
        """
        multiprocessing.set_start_method('spawn')  # we don't need the dash/webserver stuff, so spawn instead of 'fork'
        self.os_process = multiprocessing.Process(target=self.run, name=self.name)
        self.os_process.start()
        self.rpc_proxy = ServerProxy(f'http://{self.host}:{self.rpc_port}')
        return self.rpc_proxy
        
    def stop(self):
        """
        Signal the process to stop running.
        """
        print(f'[{self.name}] stopping...')
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
          
        print(f'[{self.name}] stopped')
        
    def process(self):
        """
        Perform one interation of the processing loop.
        """
        for i in range(len(self.streams.streams)):
            self.streams.streams[i].process()
