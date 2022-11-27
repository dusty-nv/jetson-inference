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
import os
import ssl
import time
import psutil
import random
import threading
import multiprocessing
import setproctitle

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.client import ServerProxy

from jetson_utils import videoSource, videoOutput, Log

from config import config, load_config, print_config
    

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
        
    def create(self, name, source):
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


# global server instance
server = None


# what to call this...
#  class WebRTCServer
#  class BackendServer
#  class InferenceServer
#  class MediaServer
class Server:
    """
    Backend media streaming server for handling assets/resources like cameras, DNN models, datasets, ect.
    It captures video from a variety of input sources (e.g. V4L2 cameras, MIPI CSI, RTP/RTSP),
    performs inferencing, and then encodes the stream and transmits it with WebRTC to the clients.

    See the Streams object and Streams.add() for how to open video devices.
    Set the Models object and Models.add() for how to load/create DNN models.
    
    It can be run in a handful of different ways:
        * process() runs one iteration of the processing loop from the calling process/thread
        * run() runs the processing loop forever in the calling process/thread 
        * start() starts a new process that it runs forever in
        * connect() attempts to connect to an existing process, and if not starts one
        * running 'python3 server.py' to launch it manually (see __main__ below)
        
    It typically runs in it's own process where it uses RPC for command & control.
    The Dash app will automatically start it, so you don't normally need to worry about it.
    """
    def __init__(self, name='server-backend', host='0.0.0.0', rpc_port=49565, webrtc_port=49567, 
                 ssl_cert=None, ssl_key=None, stun_server=None):
        """
        Create a new instance of the backend server.
        
        Parameters:
            name (string) -- name of the backend server process (also used for logging)
            host (string) -- hostname/IP of the backend server to bind/connect to
            rpc_port (int) -- port used for RPC server
            webrtc_port (int) -- port used for WebRTC server
            ssl_cert (string) -- path to PEM-encoded SSL/TLS certificate file for enabling HTTPS
            ssl_key (string) -- path to PEM-encoded SSL/TLS key file for enabling HTTPS
            stun_server (string) -- override the default WebRTC STUN server (stun.l.google.com:19302)
        """ 
        self.name = name
        self.host = host
        self.rpc_port = rpc_port
        self.webrtc_port = webrtc_port
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.streams = Streams(self)
        self.run_flag = False  # this gets set to true when initialized successfully
        self.os_process = None
        server = self
        
    def init(self):
        """
        This gets called once at the beginning of run() from within the process.
        """
        if self.os_process is not None:
            setproctitle.setproctitle(multiprocessing.current_process().name)
            Log.Verbose(f"[{self.name}] started {self.name} process (pid={self.os_process.pid})")
            
        # create the RPC server
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

    def connect(self, autostart=True, retries=10):
        """
        Attempt to connect to an existing instance of the server process.
        If one is not running, start it when autostart=True
        """
        if self.os_process is None:
            time.sleep(random.uniform(0.5, 5.0))

        if not is_process_running(self.name):
            Log.Verbose(f"[{self.name}] couldn't find existing server process running")
            if autostart:
                return self.start()
            else:
                return None
        
        # create the RPC proxy object for the caller
        self.rpc_proxy = ServerProxy(f"http://{self.host}:{self.rpc_port}")
        
        # rudimentary check to confirm the process started/initialized ok
        # TODO put this in a loop and re-try multiple times for when we start loading models/ect at server start-up
        retry_count = 0
        
        for retry_count in range(retries):
            time.sleep(0.5)
        
            try:
                if self.rpc_proxy.is_running():
                    break
                else:
                    if retry_count == retries - 1:
                        raise Exception(f"[{self.name}] failed to start running")
            except ConnectionRefusedError as error:
                Log.Verbose(f"[{self.name}] {error}")
            
            #Log.Verbose(f"[{self.name}] waiting for connection... (retry {retry_count+1} of {retries})")
          
        Log.Verbose(f"[{self.name}] {psutil.Process(os.getpid()).name()} (pid={os.getpid()}) connected to {self.name} process (pid={find_process_pid(self.name)})")
        
        server = self.rpc_proxy
        return self.rpc_proxy
            
    def start(self):
        """
        Launch the server running in a new process.
        Returns the RPC proxy object for clients to call.
        """  
        # we don't need the dash/webserver stuff, so use spawn instead of fork
        # TODO look into the memory savings/implications of this
        # https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
        # multiprocessing.set_start_method("spawn")  
    
        # start the process
        self.os_process = multiprocessing.Process(target=self.run, name=self.name, daemon=True)  # use daemon=True so process automatically exits when parent process exits
        self.os_process.start()
        time.sleep(1.0)
        
        return self.connect(autostart=False)
        
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
        If you call this from your own process, it will block and not return until the server exits.
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

    def list_resources(self):   # list_assets
        """
        Return a dict of the server's assets including streams, models, and datasets
        """
        return {
            "streams" : self.streams.get_config(),
            "models" : [],  # TODO implement models/streams
            "datasets" : []
        }
 

def is_process_running(name):
    """
    Check if there is any running process that contains the given name processName.
    """
    for proc in psutil.process_iter():
        try:
            #print(proc.name())
            if name.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False;
   
def find_process_pid(name):
    """
    Find a process PID by it's name
    """
    for proc in psutil.process_iter():
        try:
            #print(proc.name())
            if name.lower() in proc.name().lower():
                return proc.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return -1;
    
                
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", default=None, type=str, help=f"path to JSON file to load global configuration from (default is {DASH_CONFIG_FILE})")
    parser.add_argument("--name", default=None, type=str, help="Name of the backend server process to use")
    parser.add_argument("--host", default=None, type=int, help="interface for the server to use (default is all interfaces, 0.0.0.0)")
    parser.add_argument("--port", default=None, type=int, help="port used for webserver (default is 8050)")
    parser.add_argument("--rpc-port", default=None, type=int, help="port used for RPC server (default is 49565)")
    parser.add_argument("--webrtc-port", default=None, type=int, help="port used for WebRTC server (default is 49567)")
    parser.add_argument("--ssl-cert", default=None, type=str, help="path to PEM-encoded SSL/TLS certificate file for enabling HTTPS")
    parser.add_argument("--ssl-key", default=None, type=str, help="path to PEM-encoded SSL/TLS key file for enabling HTTPS")
    parser.add_argument("--stun-server", default=None, type=str, help="STUN server to use for WebRTC")
    
    args = parser.parse_args()
    
    if args.config:
        config = load_config(args.config)
       
    if args.name:
        config['server']['name'] = args.name
        
    if args.host:
        config['server']['host'] = args.host
        
    if args.rpc_port:
        config['server']['rpc_port'] = args.rpc_port

    if args.webrtc_port:
        config['server']['webrtc_port'] = args.webrtc_port
        
    if args.ssl_cert:
        config['server']['ssl_cert'] = args.ssl_cert
        
    if args.ssl_key:
        config['server']['ssl_key'] = args.ssl_key
        
    if args.stun_server:
        config['server']['stun_server'] = args.stun_server
        
    print_config(config)

    server = Server(**config['server'])
    server.start()