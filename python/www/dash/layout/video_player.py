
import json
import dash

from dash import dcc, html, Input, Output, ALL, MATCH
from dash.exceptions import PreventUpdate

from .card import create_card, card_callback

from server import Server


def create_video_player(stream):
    stream_config = Server.request(f"/streams/{stream}").json()
    stream_config['video_player'] = f"video_player_element_{stream}"
    
    #print('create_video_player')
    #print(stream_config)
    
    """
    # https://stackoverflow.com/questions/23248441/resizing-video-element-to-parent-div
    # https://stackoverflow.com/questions/4000818/scale-html5-video-and-break-aspect-ratio-to-fill-whole-site
    video_player_style={
        'position': 'absolute', 
        'right': 0, 
        'bottom': 0,
        'min-width': '50%', 
        'min-height': '50%',
        'width': 'auto', 
        'height': 'auto', 
        'z-index': -100,
        'background-size': 'cover',
        'overflow': 'hidden',
    }
    """
    video_player_style={
        'width': '100%',
        #'object-fit': 'cover',
    }
    
    children = [
        html.Video(id=stream_config['video_player'], controls=True, autoPlay=True, style=video_player_style),
        html.Div(id={'type': 'hidden_div_video_player', 'index': stream}, style={'display':'none'}),
        dcc.Store(id={'type': 'video_player_config', 'index': stream}, data=json.dumps(stream_config)),
    ]
    
    return create_card(children, title=stream, id=stream, width=6, height=14, settings_button='card-settings-stream')

    
@card_callback(Input({'type': 'navbar_stream', 'index': ALL}, 'n_clicks'))
def play_stream(n_clicks):
    #print(f"play stream {dash.ctx.triggered_id['index']}   n_clicks={n_clicks}")
    #print(f"n_clicks:  {n_clicks}")
    #print(dash.ctx.triggered)
    
    if dash.ctx.triggered[0]['value'] > 0:
        return create_video_player(dash.ctx.triggered_id['index'])
        
    return None
    

dash.clientside_callback(
    dash.ClientsideFunction('webrtc', 'playStream'),
    Output({'type': 'hidden_div_video_player', 'index': MATCH}, 'children'),  # script has no output, but dash callbacks must have outputs
    Input({'type': 'video_player_config', 'index': MATCH}, 'data'),
    #prevent_initial_call=True
)
