#!/usr/bin/env python3

from server import Server, Action
#from typing import Union, List


class BrowserAlert(Action):
    def __init__(self):
        super().__init__()
        self._min_frames = 10
        self._labels = None
        
    @property
    def min_frames(self) -> int: #-> Union[List[int], str]:
        return self._min_frames
        
    @min_frames.setter
    def min_frames(self, min_frames):
        self._min_frames = int(min_frames)
        
    @property
    def labels(self) -> str:
        return self._labels
        
    @labels.setter
    def labels(self, labels):
        self._labels = labels
    
    def on_event(self, event):
        if event.frames > self._min_frames and not hasattr(event, 'alert_triggered'):
            Server.alert(f"Detected '{event.label}' ({event.maxScore * 100:.1f}%)")
            event.alert_triggered = True
            
''' 
@action('Browser Alerts')
def on_event(event, min_frames=10):
    """
    Action that triggers a browser alert notification once per object
    """
    if event.frames > 10 and not hasattr(event, 'action_triggered'):
        Server.alert(f"Detected '{event.label}' ({event.maxScore * 100:.1f}%)")
        event.action_triggered = True
'''