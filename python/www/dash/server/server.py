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
import sys
import time
import random

import ssl
import json
import flask
import requests

import psutil
import traceback
import threading
import multiprocessing
import setproctitle

from xmlrpc.client import ServerProxy
from xmlrpc.server import SimpleXMLRPCServer

from jetson_utils import Log

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model
from stream import Stream

    
def server_request(*args, **kwargs):
    args = list(args)
    args[1] = f"{Server.instance.rest_url}{args[1]}"
    return requests.request(*args, **kwargs)
    
    
# what to call this...
#  class WebRTCServer
#  class BackendServer
#  class InferenceServer
#  class MediaServer
#  class DNNServer
class Server:
    """
    Backend media streaming server for handling resources like cameras, DNN models, datasets, ect.
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
    Once start() or connect() are run, the server instance will be replaced with
    the RPC proxy that forwards function calls to the process running the server.
    
    This class is typically a singleton and can be accessed with Server.instance 
    """
    instance = None   # singleton instance
    api = None        # flask REST server
    
    def __init__(self, name='server-backend', host='0.0.0.0', 
                 rpc_port=49565, rest_port=49566, webrtc_port=49567, 
                 ssl_cert=None, ssl_key=None, stun_server=None, resources=None):
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
            resources (string or dict) -- either a path a json config file or dict containing resources to load
        """
        Server.instance = self
        self.name = name
        self.host = host
        self.rpc_port = rpc_port
        self.rest_url = f"{'https://' if ssl_cert else 'http://'}{host}:{rest_port}"
        self.rest_port = rest_port
        self.webrtc_port = webrtc_port
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.os_process = None  
        self.run_flag = False            # this gets set to true when initialized successfully
        self.init_resources = resources  # these resources get loaded during init()
        self.resources = {
            'models': {},
            'streams' : {},
            #'datasets': {},
        }
        
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

        # create the rest server
        Server.api = flask.Flask(__name__)
        self.api_thread = threading.Thread(target=lambda: Server.api.run(host=self.host, port=self.rest_port,
                                           ssl_context=(self.ssl_cert, self.ssl_key) if self.ssl_cert else None),
                                           name=f"{self.name}-rest")
        self.api_thread.start()
        
        # register rest API's
        Server.api.add_url_rule('/status', view_func=self.get_status, methods=['GET'])
        Server.api.add_url_rule('/resources', view_func=self.get_resources, methods=['GET'])
        
        Log.Info(f"[{self.name}] REST server is running @ {self.rest_url}")
        
        # load resources
        self.load_resources(self.init_resources)
        
        # indicate that server is ready to run
        self.run_flag = True
        
    def connect(self, autostart=True, retries=10):
        """
        Attempt to connect to an existing instance of the server process.
        If one is not running, start it when autostart=True
        """
        if self.os_process is None:
            time.sleep(random.uniform(0.5, 5.0))

        if autostart and not is_process_running(self.name):
            Log.Verbose(f"[{self.name}] couldn't find existing server process running")
            return self.start()
        
        # run a rest status query
        retry_count = 0
        
        for retry_count in range(retries):
            time.sleep(0.5)
            
            try:
                response = Server.request('GET', '/status')
                print(f'server status:  {response.json()}')
                if response.ok and response.json()['running']:
                    break
                else:
                    if retry_count == retries - 1:
                        raise Exception(f"[{self.name}] failed to start running")
            except Exception as error:
                traceback.print_exc()
                
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
        
        Server.instance = self.rpc_proxy
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
        Log.Info(f"[{self.name}] stopping...")
        
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
          
        Log.Info(f"[{self.name}] stopped")
        
    def is_running(self):
        """
        Returns true if the process is initialized and running.
        """
        return self.run_flag
        
    def process(self):
        """
        Perform one interation of the processing loop.
        """
        if len(self.resources['streams']) > 0:
            for stream in self.resources['streams'].values():
                stream.process()
        else:
            time.sleep(1.0)

    def add_resource(self, group, name, *args, **kwargs):
        """
        Add a resource to the server.
        
        Parameters:
            group (string) -- should be one of:  'streams', 'models', 'datasets'
            name (string)  -- the name of the resource
            args (list)    -- arguments to create the resource with
        """
        if group not in self.resources:
            Log.Error(f"[{self.name}] invalid resource group '{group}'")
            return
            
        try:
            if group == 'models':
                resource = Model(self, name, *args, **kwargs)
            elif group == 'streams':
                resource = Stream(self, name, *args, **kwargs)
            else:
                Log.Error(f"[{self.name}] invalid resource group '{group}' for resource '{name}'")
                return
        except Exception as error:
            Log.Error(f"[{self.name}] failed to create resource '{name}' in group '{group}'")
            traceback.print_exc()
            return
        
        self.resources[group][name] = resource
        return resource.get_config()
    
    def get_resource(self, group, name):
        """
        Return a config dict of a resource from a particular group.
        
        Parameters:
            group (string) -- should be one of:  'streams', 'models', 'datasets'
            name (string)  -- the name of the resource
        """
        return self.resources[group][name].get_config()
        
    def list_resources(self, groups=None):
        """
        Return a config dict from a group or groups of the server's resources.
        By default, resources from all of the groups will be returned (models, streams, and datasets).
        If the requested group is a string, only resources from that group will be returned.
        If the requested group is a list, resources from each of those groups will be returned.
        """ 
        if groups is None:
            groups = self.resources.keys()
        elif isinstance(groups, str):
            return { name : resource.get_config() for (name, resource) in self.resources[groups].items() }
            
        resources = {}
        
        for group in groups:
            resources[group] = { name : resource.get_config() for (name, resource) in self.resources[group].items() }
 
        return resources
 
    def load_resources(self, resources):
        """
        Load resources (streams/models/datasets) from a json config file or dict
        
        Parameters:
            resources (string or dict) -- a path to a json config file, or a dict
                                          containing a representation of the resource
        """
        if resources is None:
            return
            
        if isinstance(resources, str):
            if not os.path.exists(resources):
                Log.Error(f"[{self.name}] path does not exist: {resources}")
                return
              
            Log.Info(f"[{self.name}] loading resources from {resources}")
            
            with open(resources) as file:
                resources = json.load(file)
            
        if not isinstance(resources, dict):
            Log.Error(f"[{self.name}] load_resources() must be called with a string or dict")
            return
        
        for group in self.resources.keys():
            if group not in resources:
                continue
                
            for name, resource in resources[group].items():
                self.add_resource(group, name, **resource)
         
    @classmethod
    def request(cls, *args, **kwargs):
        """
        Wrapper around requests.request() that appends the server's address to the request URL.
        This can be used to make JSON REST API requests to the server.
        """
        args = list(args)
        args[1] = f"{Server.instance.rest_url}{args[1]}"
        return requests.request(*args, **kwargs)
        
    def get_status(self):
        """
        /status REST API
        """
        Log.Verbose(f"[{self.name}] GET /status REST API")
        return flask.jsonify({'running': self.is_running()})
        
    def get_resources(self):
        """
        /resources REST API
        """
        Log.Verbose(f"[{self.name}] GET /resources REST API")
        return flask.jsonify(self.list_resources())
        
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
    from config import config, load_config, print_config
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", default=None, type=str, help=f"path to JSON file to load global configuration from")
    parser.add_argument("--name", default=None, type=str, help="Name of the backend server process to use")
    parser.add_argument("--host", default=None, type=int, help="interface for the server to use (default is all interfaces, 0.0.0.0)")
    parser.add_argument("--port", default=None, type=int, help="port used for webserver (default is 8050)")
    parser.add_argument("--rpc-port", default=None, type=int, help="port used for RPC server (default is 49565)")
    parser.add_argument("--webrtc-port", default=None, type=int, help="port used for WebRTC server (default is 49567)")
    parser.add_argument("--ssl-cert", default=None, type=str, help="path to PEM-encoded SSL/TLS certificate file for enabling HTTPS")
    parser.add_argument("--ssl-key", default=None, type=str, help="path to PEM-encoded SSL/TLS key file for enabling HTTPS")
    parser.add_argument("--stun-server", default=None, type=str, help="STUN server to use for WebRTC")
    parser.add_argument("--resources", default=None, type=str, help="path to JSON config file to load initial server resources from")
    parser.add_argument("--connect", action="store_true", help="connect to the server instead of starting it")
    
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
 
    if args.resources:
        config['server']['resources'] = args.resources
        
    print_config(config)

    server = Server(**config['server'])
    
    if args.connect:
        server.connect(autostart=False)
    else:
        server.run()