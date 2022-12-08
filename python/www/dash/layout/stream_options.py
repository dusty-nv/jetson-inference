
import json
import dash

from dash import dcc, html, Input, Output, ALL, MATCH
from dash.exceptions import PreventUpdate

from .card import create_card, card_callback

from server import Server


def create_stream_options(stream):
    stream_config = Server.instance.get_resource('streams', stream)
    
    """
    children = [
        html.Video(id=stream_config['video_player'], controls=True, autoPlay=True, style=video_player_style),
        html.Div(id={'type': 'hidden_div_video_player', 'index': stream}, style={'display':'none'}),
        dcc.Store(id={'type': 'video_player_config', 'index': stream}, data=json.dumps(stream_config)),
    ]
    """
    
    return create_card(f"Stream Settings for {stream}", title=f"{stream} Settings", id=f"stream_options_{stream}")
        
        
@card_callback(Input({'type': 'card-settings-stream', 'index': ALL}, 'n_clicks'))
def on_stream_settings(n_clicks):
    if dash.ctx.triggered[0]['value'] > 0:
        return create_stream_options(dash.ctx.triggered_id['index'])
