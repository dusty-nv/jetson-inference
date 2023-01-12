#!/usr/bin/env python3

from server import Server, Action, EventFilter


class BrowserAlert(Action, EventFilter):
    """
    Action that triggers browser alerts and supports event filtering.
    """
    def __init__(self):
        super(BrowserAlert, self).__init__()
        
        self._test_bool = False
        self._test_float = 20.0
    
    @property
    def test_bool(self) -> bool:
        return self._test_bool
        
    @test_bool.setter
    def test_bool(self, value):
        self._test_bool = value
        
    @property
    def test_float(self) -> float:
        return self._test_float

    @test_float.setter
    def test_float(self, value):
        self._test_float = value
        
    def on_event(self, event):
        if self.filter(event) and not hasattr(event, 'alert_triggered'):
            Server.alert(f"Detected '{event.label}' ({event.maxScore * 100:.1f}%)")
            event.alert_triggered = True
