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

import copy


class Action:
    """
    Base class for actions that filter events and trigger notifications/responses/ect.
    Users should inherit from this class and implement their own logic in on_event()
    Any @property attributes are automatically configurable from the webpage UI.
    """
    def __init__(self, name=None, enabled=False):
        self.id = -1      # gets assigned by the server (int)
        self.type = None  # gets assigned by the server (ActionType)
        self.name = name 
        self.enabled = enabled

    def on_event(self, event):
        pass
 
    def to_dict(self):
        config = {
            'id': self.id,
            'name': self.name,
            'type': self.type['name'],
            'enabled': self.enabled,
            'properties': copy.deepcopy(self.type['properties'])
        }
        
        for property in config['properties'].values():
            property['value'] = property['object'].fget(self)
            
        return config
        
        
"""     
class ActionFilter:
    
    @property
    def labels(
    
    @property
    def min_frames(self):
        return self._min_frames
        
    @min_frames.setter
    def min_frames(self, min_frames):
        self._min_frames = int(min_frames)
"""
    
''' 
def action(function=None, name=None, enabled=True):
    """
    Decorator for registering an event callback that filters events and performs actions.
    Event callbacks receive one argument - an event (see below for an example) - and it
    will be called every time an event is created or updated in the system.
    
    @action
    def on_event(event):
        if event.label == 'person' and event.frames > 15:
            # do something
            
    The event objects are mutable and actions can store state in them if desired.
    """
    from server import Server
    
    if isinstance(function, str) and not name:
        name = function
       
    def register(function, name, enabled):
        key = f"{function.__module__}.{function.__name__}"
        
        if not name:
            name = key
            
        Server.instance.actions[key] = {
            'function': function,
            'enabled': enabled,
            'name': name,
        }

        return function
        
    def outer(function):
        return register(function, name, enabled)
        
    if callable(function):
        return register(function, name, enabled)
    else:   
        return outer
'''       