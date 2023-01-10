#!/usr/bin/env python3

from server import Server, action


@action('Browser Alerts')
def on_event(event):
    """
    Action that triggers a browser alert notification once per object
    """
    if event.frames > 10 and not hasattr(event, 'action_triggered'):
        Server.alert(f"Detected '{event.label}' ({event.maxScore * 100:.1f}%)")
        event.action_triggered = True