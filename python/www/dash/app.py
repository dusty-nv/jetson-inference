#!/usr/bin/env python3
#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the 'Software'),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import dash
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import dash_auth

from dash import dcc, Input, Output
from dash.exceptions import PreventUpdate

from config import config, print_config
from server import server, Server

from layout.grid import create_grid
from layout.navbar import create_navbar
from layout.add_stream import add_stream_dialog

from layout.video_player import create_video_player
#from layout.test_card import create_test_card

#import os
#print(f'loaded {__file__} module from {__name__} (pid {os.getpid()})')

# create the dash app
external_scripts = ['https://webrtc.github.io/adapter/adapter-latest.js']
external_stylesheets = [] #[dbc.themes.DARKLY] #[dbc.themes.SLATE] #[dbc.themes.FLATLY] #[dbc.themes.SUPERHERO] #['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, title=config['dash']['title'],
                external_scripts=external_scripts, 
                external_stylesheets=external_stylesheets, 
                suppress_callback_exceptions=True)
                
webserver = app.server

if len(config['dash']['users']) > 0:
    auth = dash_auth.BasicAuth(app, config['dash']['users'])

app.layout = dash.html.Div([
    create_navbar(),
    add_stream_dialog(),
    create_grid(),
    dcc.Store(id='resources_config'),
    dcc.Interval(id='refresh_timer', interval=config['dash']['refresh'])
])

@app.callback(Output('resources_config', 'data'),
              Input('refresh_timer', 'n_intervals'),
              Input('resources_config', 'data'))
def on_refresh(n_intervals, previous_config):
    """
    Get the latest resources config from the server.
    This can trigger updates to the clientside nav structure.
    """
    #print(f'refresh config {n_intervals}')
    resources_config = server.list_resources()
    #print(f'server config:  ({len(server_config["streams"])} streams)\n{server_config}')
    
    if previous_config is not None:
        if resources_config == previous_config:
            raise PreventUpdate   # if the config hasn't changed, skip the update

    print(f'received updated config from backend server')
    return resources_config
    
    #print(f'received updated config from backend server')
    #return [create_navbar(server_config), json.dumps(server_config)]
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(epilog="see config.py & data/config.json for more settings")
    
    parser.add_argument("--host", default=None, type=str, help="interface for the webserver to use (default is all interfaces, 0.0.0.0)")
    parser.add_argument("--port", default=None, type=int, help="port used for webserver (default is 8050)")
    
    args = parser.parse_args()

    if args.host:
        config['dash']['host'] = args.host
        
    if args.port:
        config['dash']['port'] = args.port

    print_config(config)
    
    # check if HTTPS/SSL requested
    ssl_context = None
    
    if config['server']['ssl_cert'] and config['server']['ssl_key']:
        ssl_context = (config['server']['ssl_cert'], config['server']['ssl_key'])
        
    # start the backend server
    server = Server(**config['server'])
    server = server.start()
    
    # disable code reloading because it starts the app multiple times (https://dash.plotly.com/devtools)
    # https://community.plotly.com/t/dash-multi-page-app-functions-are-called-twice-unintentionally/46450
    app.run_server(host=config['dash']['host'], port=config['dash']['port'], ssl_context=ssl_context, debug=True, use_reloader=False)

else:
    # gunicorn instance
    # start the backend server
    server = Server(**config['server'])
    server = server.connect()
    
"""   
import argparse
import json

import dash
import dash_bootstrap_components as dbc

from dash import Dash, dcc, html, Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate

from server import Server
from users import Authenticate

external_scripts = ['https://webrtc.github.io/adapter/adapter-latest.js']
external_stylesheets = [] #[dbc.themes.DARKLY] #[dbc.themes.SLATE] #[dbc.themes.FLATLY] #[dbc.themes.SUPERHERO] #['https://codepen.io/chriddyp/pen/bWLwgP.css']

app_title = 'WebRTC Dashboard'  # make this configurable

app = Dash(__name__, external_scripts=external_scripts, external_stylesheets=external_stylesheets, title=app_title)
auth = Authenticate(app)


def create_navbar(config=[]):
    # dynamically create navbar children based on the current server config
    if len(config) == 0 or len(config['streams']) == 0:   # empty config / default navbar
        return [dbc.NavLink('Add Stream', id='navbar_add_stream')]
        
    # populate streams menu
    stream_menu_items = [
        dbc.DropdownMenuItem('Add Stream', id='navbar_add_stream', n_clicks=0),
        dbc.DropdownMenuItem('Streams', header=True),
    ] 
    
    stream_menu_items += [
        dbc.DropdownMenuItem(name, id={'type': 'navbar_stream', 'index': name}) for name in config['streams']
    ]
        
    return [dbc.DropdownMenu(
        children=stream_menu_items,
        nav=True,
        in_navbar=True,
        label='Streams',
        id='navbar_streams_dropdown')]
        
navbar = dbc.NavbarSimple(
    children=create_navbar(),
    brand=app_title,
    #brand_href='#',
    color='primary',
    id='navbar',
    dark=True,
)

add_stream_dialog = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle('Add Stream')),
        dbc.ModalBody(dbc.Form([
            html.Div([
                dbc.Label('Stream Name', html_for='add_stream_name'),
                dbc.Input(id='add_stream_name', value='/my_stream'), #placeholder='/my_stream'),
                dbc.FormText('Your name of the stream (e.g. location)'),
            ], className='mb-3'),
            html.Div([
                dbc.Label('Video Source', html_for='add_stream_source'),
                dbc.Input(id='add_stream_source', value='/dev/video0'), #placeholder='/dev/video0'),
                dbc.FormText('Camera input: /dev/video*, csi://0, rtsp://, file://'),
            ], className='mb-3'),
            html.Div([
                dbc.Checklist(
                    options=[{'label' : 'Auto Play', 'value': 'auto_play'}],
                    value=['auto_play'],
                    id='add_stream_checklist'),
            ], className='mb-3'),
        ])),
        dbc.ModalFooter(
            dbc.Button('Create', id='add_stream_submit', className='ms-auto', n_clicks=0)
        )],
    id='add_stream_dialog',
    is_open=False)
        
@app.callback(
    Output('add_stream_dialog', 'is_open'),
    Input('navbar_add_stream', 'n_clicks'), 
    Input('add_stream_submit', 'n_clicks'),
    State('add_stream_dialog', 'is_open'),
)
def show_add_stream_dialog(n1, n2, is_open):
    if n1 == 0: 
        raise PreventUpdate

    if n1 or n2:
        return not is_open
    return is_open
    
app.layout = html.Div([
    navbar,
    add_stream_dialog,
    html.Video(id='video_player', controls=True, autoPlay=True),
    html.Div(id='status_text'),
    html.Div(id='hidden_div', style={'display':'none'}),    # used as null output for playStream() javascript callback
    dcc.Store(id='play_stream_config'),                     # used to trigger a stream being played in the video player
    dcc.Store(id='play_stream_config_add'),                 # when a stream gets added, it triggers the callback chain with this
    dcc.Store(id='server_config'),                          # used to store the server streams config the UI is based on
    dcc.Interval(id='refresh_nav_timer', interval=1000)     # timer that pulls the streams config from the server and refreshes the UI
])

@app.callback(Output('status_text', 'children'),
              Output('play_stream_config_add', 'data'),
              Input('add_stream_submit', 'n_clicks'),
              State('add_stream_name', 'value'),
              State('add_stream_source', 'value'),
              State('add_stream_checklist', 'value'),
              prevent_initial_call=True)
def add_stream(n_clicks, name, source, options):
    stream_config = server.streams.add(name, source)
    return [
        html.Div([
            f'Adding stream:  {name}', html.Br(), 
            f'Stream source:  {source}', html.Br(), 
            f'{stream_config}', html.Br()]),
        json.dumps(stream_config) if 'auto_play' in options else dash.no_update]
    

@app.callback(
    Output('play_stream_config', 'data'),
    Input({'type': 'navbar_stream', 'index': ALL}, 'n_clicks'),
    Input('play_stream_config_add', 'data'),
    State('server_config', 'data'))
def play_stream(n_clicks, added_stream, server_config):
    if isinstance(dash.ctx.triggered_id, str) and dash.ctx.triggered_id == 'play_stream_config_add':
        return added_stream  # the 'Add Stream' dialog triggered this
    elif not isinstance(dash.ctx.triggered_id, dict): 
        raise PreventUpdate  # only the navbar buttons make dict id's
    
    if n_clicks[0] is None:
        raise PreventUpdate
        
    stream_name = dash.ctx.triggered_id['index']
    server_config = json.loads(server_config)
    
    return json.dumps(server_config['streams'][stream_name])

app.clientside_callback(
    dash.ClientsideFunction('webrtc', 'playStream'),
    Output('hidden_div', 'children'),  # script has no output, but dash callbacks must have outputs
    Input('play_stream_config', 'data'),
    prevent_initial_call=True
)
    
@app.callback(Output('navbar', 'children'),
              Output('server_config', 'data'),
              Input('refresh_nav_timer', 'n_intervals'),
              Input('server_config', 'data'))
def refresh_nav(n_intervals, previous_config_json):
    #print(f'refresh config {n_intervals}')
    server_config = server.get_config()
    #print(f'server config:  ({len(server_config["streams"])} streams)\n{server_config}')
    
    if previous_config_json is not None:
        previous_config = json.loads(previous_config_json)
        
        if server_config == previous_config:
            raise PreventUpdate   # if the config hasn't changed, skip the update

    print(f'received updated config from backend server')
    return [create_navbar(server_config), json.dumps(server_config)]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--host', default='0.0.0.0', type=str)
    parser.add_argument('--port', default=8050, type=int)
    parser.add_argument('--rpc-port', default=49565, type=int)
    parser.add_argument('--webrtc-port', default=49567, type=int)
    parser.add_argument('--ssl-cert', default='', type=str)
    parser.add_argument('--ssl-key', default='', type=str)
    #parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()

    # check if HTTPS/SSL requested
    ssl_context = None
    
    if args.ssl_cert and args.ssl_key:
        ssl_context = (args.ssl_cert, args.ssl_key)
        print(f'using SSL cert: {args.ssl_cert}')
        print(f'using SSL key:  {args.ssl_key}')
        
    # start the backend server
    server = Server(host=args.host, rpc_port=args.rpc_port, webrtc_port=args.webrtc_port, ssl_context=ssl_context)
    server = server.start()
    
    # disable code reloading because it starts the app multiple times (https://dash.plotly.com/devtools)
    # https://community.plotly.com/t/dash-multi-page-app-functions-are-called-twice-unintentionally/46450
    app.run_server(host=args.host, port=args.port, ssl_context=ssl_context, debug=True, use_reloader=False)
    
    server.stop()
    print('exiting app')
"""  