#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import dash
import dash_bootstrap_components as dbc

from dash import dcc, html, Input, Output, State, MATCH
from dash.exceptions import PreventUpdate

from server import Server


def create_actions_dialog():
    """
    Create the top-level dialog container used for configuring actions.
    It's children will be created dynamically when the dialog is opened.
    """
    children = [
        dbc.ModalHeader(dbc.ModalTitle('Actions')),
        dbc.ModalBody([], id='actions_dialog_body'),
        dbc.ModalFooter(dbc.Button('Close', id='actions_close', className='ms-auto', n_clicks=0)),
    ]
    
    return dbc.Modal(children, id='actions_dialog', is_open=False)
    
    
def create_actions_body():
    """
    Create the dialog body used for configuring actions.
    """
    actions = Server.request('/actions').json()
    children = []
    
    def action_name(key, name):
        if key == name:  return key
        else:  return f"{name} ({key})"
        
    for key, action in actions.items():
        children += [
            dbc.Checklist(
                options = [{'label': action_name(key, action['name']), 'value': key}],
                value = [key] if action['enabled'] else [],
                id = {'type': 'action_switch', 'index': key},
                switch = True),
            html.Div(id={'type': 'hidden_div_action', 'index': key}, style={'display':'none'})
        ]
    
    return children
    
@dash.callback(
    Output('actions_dialog', 'is_open'),
    Output('actions_dialog_body', 'children'),
    Input('navbar_actions', 'n_clicks'), 
    Input('actions_close', 'n_clicks'),
    State('actions_dialog', 'is_open'),
)
def show_actions_dialog(n1, n2, is_open):
    """
    Callback for triggering to show/hide the actions dialog
    """
    print(f'show_actions_dialog({n1}, {n2}, {is_open})')
    print(dash.ctx.triggered)

    if not dash.ctx.triggered[0]['value']:
        raise PreventUpdate

    if is_open:
        return False, dash.no_update
    else:
        return True, create_actions_body()
   
 
@dash.callback(
    Output({'type': 'hidden_div_action', 'index': MATCH}, 'children'),
    Input({'type': 'action_switch', 'index': MATCH}, 'value'),
)
def on_action_toggled(value):
    if not dash.ctx.triggered_id:
        raise PreventUpdate
        
    Server.request('PUT', f"/actions/{dash.ctx.triggered_id['index']}", json={'enabled': len(value) > 0})
    raise PreventUpdate
    
    
"""
@dash.callback(
    Output('hidden_div_stream', 'children'),
    Input('stream_options_submit', 'n_clicks'),
    State('stream_options_name', 'value'),
    State('stream_options_source', 'value'),
    State('stream_options_model', 'value'),
)
def stream_submit(n_clicks, name, source, model):
    if n_clicks == 0:
        raise PreventUpdate
        
    print(f"adding stream {name} from source {source} with model {model}")
    Server.request('POST', '/streams', json={'name': name, 'source': source, 'models': model})
    raise PreventUpdate
"""
