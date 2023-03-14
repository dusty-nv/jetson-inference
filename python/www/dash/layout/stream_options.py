
import dash
import dash_bootstrap_components as dbc

from dash import dcc, html, callback, Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from server import Server


def create_stream_dialog(stream={}):
    """
    Create the top-level dialog container used for creating/configuring streams.
    It's children will be created dynamically in create_stream_options() below.
    """
    return dbc.Modal(create_stream_options(stream), id='stream_options_dialog', is_open=False)
    
    
def create_stream_options(stream={}):
    """
    Create the dialog body used for creating/configuring streams.
    """
    children = [dbc.ModalHeader(dbc.ModalTitle(stream.get('name', 'Add Stream')))]
    
    form = dbc.Form([
        html.Div([
            dbc.Label('Stream Name', html_for='stream_options_name'),
            dbc.Input(id='stream_options_name', value='/my_stream'), #placeholder='/my_stream'),
            dbc.FormText('Your name of the stream (e.g. location)'),
        ], className='mb-3'),
        
        html.Div([
            dbc.Label('Video Source', html_for='stream_options_source'),
            dbc.Input(id='stream_options_source', value='/dev/video0'), #placeholder='/dev/video0'),
            dbc.FormText('Camera input: /dev/video*, csi://0, rtsp://, file://'),
        ], className='mb-3'),
        
        html.Div([
            dbc.Label('Models', html_for='stream_options_model'),
            #dbc.Select(options=list_models(), multi=True, id='stream_options_model'),
            dcc.Dropdown(options=list_models(), multi=True, id='stream_options_model'),
            dbc.FormText("The DNN model(s) to use for processing the stream"),
        ], className='mb-3'),
        
        html.Div([
            dbc.Checklist(
                options=[{'label' : 'Auto Play', 'value': 'auto_play'}],
                value=['auto_play'],
                id='stream_options_checklist'),
        ], className='mb-3'),
        
        html.Div(id='hidden_div_stream', style={'display':'none'})
    ])

    children += [dbc.ModalBody(form)]
    children += [dbc.ModalFooter(dbc.Button('Create', id='stream_options_submit', className='ms-auto', n_clicks=0))]
    
    return children


def list_models():
    """
    Return a drop-down list of models from the server that can be selected.
    """
    models = Server.request('/models').json() if Server.instance is not None else []
    return [{'label': model, 'value': model} for model in models]
    
    
@dash.callback(
    Output('stream_options_dialog', 'is_open'),
    Output('stream_options_dialog', 'children'),
    Input('navbar_add_stream', 'n_clicks'), 
    Input('stream_options_submit', 'n_clicks'),
    Input({'type': 'card-settings-stream', 'index': ALL}, 'n_clicks'),
    State('stream_options_dialog', 'is_open'),
)
def show_stream_dialog(n1, n2, n3, is_open):
    """
    Callback for triggering to show/hide the stream options dialog
    """
    stream = {}
    
    #print(f'show_stream_dialog({n1}, {n2}, {n3}, {is_open})')
    #print(dash.ctx.triggered)

    if not dash.ctx.triggered[0]['value']:
        raise PreventUpdate

    if isinstance(dash.ctx.triggered_id, dict) and dash.ctx.triggered_id['type'] == 'card-settings-stream':
        stream = Server.request(f"/streams/{dash.ctx.triggered_id['index']}").json()
        
    if is_open:
        return False, dash.no_update
    else:
        return True, create_stream_options(stream)
   
   
@dash.callback(
    Output('hidden_div_stream', 'children'),
    Input('stream_options_submit', 'n_clicks'),
    State('stream_options_name', 'value'),
    State('stream_options_source', 'value'),
    State('stream_options_model', 'value'),
)
def stream_submit(n_clicks, name, source, model):
    """
    Callback for when the stream form is submitted
    """
    if n_clicks == 0:
        raise PreventUpdate
        
    print(f"adding stream {name} from source {source} with model {model}")
    Server.request('POST', '/streams', json={'name': name, 'source': source, 'models': model})
    raise PreventUpdate
    
'''    
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
'''