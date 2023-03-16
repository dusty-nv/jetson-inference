#!/usr/bin/env python3

from server import Server, Action, EventFilter


class BrowserAlert(Action, EventFilter):
    """
    Action that triggers browser alerts and supports event filtering.
    """
    def __init__(self):
        super().__init__()

    def on_event(self, event):
        if self.filter(event) and not hasattr(event, 'alert_triggered'):
            Server.alert(f"Detected '{event.label}' ({event.maxScore * 100:.1f}%)")
            event.alert_triggered = True
