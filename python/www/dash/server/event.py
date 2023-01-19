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
        self.scores = [(self.begin,score)]
        
        Server.instance.events.append(self)
        self.dispatch()
                    
    def update(self, score):
        """
        Update an event with new results
        """
        self.end = time()
        self.score = score
        self.maxScore = max(self.maxScore, score)
        self.scores.append((self.end, score))
        self.frames += 1
        self.dispatch()
        
    def dispatch(self):
        """
        Send this event to actions for processing
        """
        for action in Server.instance.actions:
            if action.enabled:
                try:
                    action.on_event(self)
                except Exception as error:
                    Log.Error(f"[{Server.instance.name}] failed to run action {action.name}")
                    traceback.print_exc()
        
    def to_dict(self):
        """
        Return a dict representation of the event
        """
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
            'maxScore': self.maxScore,
            'scores': self.scores,
        }
      
    def to_list(self):
        """
        Return a list representation of the event
        """
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
            self.maxScore,
            self.scores
        ]
