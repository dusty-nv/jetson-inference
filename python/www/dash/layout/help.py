
import dash

from dash import html, Input, ALL
from .card import create_card, card_callback


def create_help(id='help'):
    return create_card(
        # github.com pages can't be embedded directly because of their CSP frame-ancestors policy
        html.Iframe(src="https://nvidia-ai-iot.github.io/jetson-min-disk/", style={'height':'100%'}, id='help_iframe'), 
        title=f"Help", 
        width=6,
        height=20,
        id=id
    )

"""
@card_callback(Input('navbar_help', 'n_clicks'))
def open_help(n_clicks):
    if n_clicks > 0:
        return create_help()  
    else:
        return None
"""