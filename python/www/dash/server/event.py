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

from time import time
from server import Server

import traceback


class Event:
    """
    Represents a classification/detection event.
    """
    def __init__(self, stream, model, classID, label, score):
        """
        Create a new event
        """
        self.id = len(Server.instance.events)
        self.stream = stream
        self.model = model
        self.classID = classID
        self.label = label
        self.score = score
        self.maxScore = score
        
        self.begin = time()
        self.end = self.begin
        self.frames = 0
        
        Server.instance.events.append(self)
        self.dispatch()
                    
    def update(self, score):
        self.score = score
        self.maxScore = max(self.maxScore, score)
        self.end = time()
        self.frames += 1
        self.dispatch()
        
    def dispatch(self):
        from server import Server
        for action in Server.instance.actions:
            if action.enabled:
                try:
                    action.on_event(self)
                except Exception as error:
                    Log.Error(f"[{Server.instance.name}] failed to run action {action.name}")
                    traceback.print_exc()
        
    def to_dict(self):
        return {
            'id': self.id,
            'begin': self.begin,
            'end': self.end,
            'frames': self.frames,
            'stream': self.stream.name,
            'model': self.model.name,
            'classID': self.classID,
            'label': self.label,
            'score': self.score,
            'maxScore': self.maxScore
        }
      
    def to_list(self):
        return [
            self.id,
            self.begin,
            self.end,
            self.frames,
            self.stream.name,
            self.model.name,
            self.classID,
            self.label,
            self.score,
            self.maxScore
        ]


class EventFilter:
    """
    Class for filtering events.  Inherit your actions from this class to automatically
    add filtering properties and call `self.filter(event)` in your `on_event()` callback.
    """
    def __init__(self, labels=[], min_frames=None, **kwargs):
        """
        Initialize a new filter.
        """
        super(EventFilter, self).__init__()
        self._labels = labels
        self._min_frames = min_frames
        
    def filter(self, event):
        """
        Return true if an event passes the filter, otherwise false.
        """
        if len(self._labels) > 0 and event.label not in self._labels:
            return False
            
        if self._min_frames and event.frames < self._min_frames:
            return False
            
        return True
        
    @property
    def labels(self) -> str:
        return ';'.join(self._labels)
        
    @labels.setter
    def labels(self, labels):
        if isinstance(labels, str):
            self._labels = labels.split(';')
            self._labels = [label.strip() for label in self._labels]
        else:
            self._labels = labels
        
    @property
    def min_frames(self) -> int:
        return self._min_frames
        
    @min_frames.setter
    def min_frames(self, min_frames):
        self._min_frames = int(min_frames)
        