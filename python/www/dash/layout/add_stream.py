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
import dash_bootstrap_components as dbc

from dash import dcc, html, callback, Input, Output, State
from dash.exceptions import PreventUpdate


def add_stream_dialog(title='Add Stream'):
    """
    Create a dialog used for add streams.
    """
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(title)),
        dbc.ModalBody(
            dbc.Form([
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
            ])
        ),
        dbc.ModalFooter(dbc.Button('Create', id='add_stream_submit', className='ms-auto', n_clicks=0)),
    ],
    id='add_stream_dialog',
    is_open=False)

   
@dash.callback(
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
   
"""
@dash.callback(
    Output('play_stream_config_add', 'data'),
    Input('add_stream_submit', 'n_clicks'),
    State('add_stream_name', 'value'),
    State('add_stream_source', 'value'),
    State('add_stream_checklist', 'value'),
    prevent_initial_call=True)
def on_add_stream(n_clicks, name, source, options):
    stream_config = server.streams.add(name, source)
    return [
        html.Div([
            f'Adding stream:  {name}', html.Br(), 
            f'Stream source:  {source}', html.Br(), 
            f'{stream_config}', html.Br()]),
        json.dumps(stream_config) if 'auto_play' in options else dash.no_update]
"""