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
    Create the dialog body used for creating/configuring actions.
    """
    action_types = Server.request('/actions/types').json()
    
    print('ACTION_TYPES:')
    print(action_types)

    children = [
        html.Div([
            dbc.Select( 
                id='create_action_type',
                placeholder="Create new action...",
                options=[{'label': f"{type['class']} ({type['name']})", 'value': type['name']} for type in action_types.values()],
                style={'flexGrow': 100},
            ),
            dbc.Button('Create', id='create_action_button', className='ms-2', n_clicks=0),
        ], style={'display': 'flex'}),
        html.Hr(),
        html.Div(create_action_settings(), id='action_settings')
    ]
    
    return children
   
   
def create_action_settings():
    """
    Create components for configuring each action
    """
    actions = Server.request('/actions').json()
    children = []
    
    print('ACTIONS:')
    print(actions)
    
    for action in actions:
        children += [
            dbc.Switch(
                id={'type': 'action_enabled', 'index': action['id']}, 
                label=f"[{action['id']}] {action['name']}",
                value=action['enabled']
            ),
            html.Div(id={'type': 'hidden_div_action', 'index': action['id']}, style={'display':'none'})
        ]
        
        for property_name, property in action['properties'].items():
            index = f"{action['id']}.{property_name}"
            disabled = not property['mutable']
            debounce = True
            
            if property['type'] == 'str' or property['type'] is None:
                control = dbc.Input(
                    type='text', 
                    value=property['value'], 
                    disabled=disabled, 
                    debounce=debounce,
                    id={'type': 'action_property_str', 'index': index},
                )
            elif property['type'] == 'int':
                control = dbc.Input(
                    type='number', 
                    value=property['value'], 
                    disabled=disabled, 
                    debounce=debounce,
                    id={'type': 'action_property_int', 'index': index},
                )
            elif property['type'] == 'float':
                control = dbc.Input(
                    type='number', 
                    value=property['value'],
                    step=0.1,
                    disabled=disabled,
                    debounce=debounce,
                    id={'type': 'action_property_float', 'index': index},
                )   
            else:
                print(f"[dash]   warning -- skipping unsupported action property type '{property['int']}'")
                continue

            children += [
                dbc.Row([dbc.Col(property_name, width=3), dbc.Col(control)], align='center', className='mb-2'),
                html.Div(id={'type': f"hidden_div_action_{property['type']}", 'index': index}, style={'display':'none'})
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
    if not dash.ctx.triggered[0]['value']:
        raise PreventUpdate

    if is_open:
        return False, dash.no_update
    else:
        return True, create_actions_body()
   
 
@dash.callback(
    Output('action_settings', 'children'),
    Input('create_action_button', 'n_clicks'),
    State('create_action_type', 'value')
)
def on_create_action(n_clicks, value):
    if not n_clicks or not value:
        raise PreventUpdate
    
    Server.request('POST', f"/actions", json={'type': value})
    return create_action_settings()
    
    
@dash.callback(
    Output({'type': 'hidden_div_action', 'index': MATCH}, 'children'),
    Input({'type': 'action_enabled', 'index': MATCH}, 'value'),
)
def on_action_setting(enabled):
    if not dash.ctx.triggered_id:
        raise PreventUpdate
        
    Server.request('PUT', f"/actions/{dash.ctx.triggered_id['index']}", json={'enabled': enabled})
    raise PreventUpdate
    
    
@dash.callback(
    Output({'type': 'hidden_div_action_int', 'index': MATCH}, 'children'),
    Input({'type': 'action_property_int', 'index': MATCH}, 'value')
)
def on_action_property_int(value):
    """
    Callback for updating int properties
    """
    if not dash.ctx.triggered_id:
        raise PreventUpdate
        
    print(f"on_action_property_int({value}) => {dash.ctx.triggered_id}")
    raise PreventUpdate
    
    
@dash.callback(
    Output({'type': 'hidden_div_action_float', 'index': MATCH}, 'children'),
    Input({'type': 'action_property_float', 'index': MATCH}, 'value')
)
def on_action_property_float(value):
    """
    Callback for updating float properties
    """
    if not dash.ctx.triggered_id:
        raise PreventUpdate
        
    print(f"on_action_property_float({value}) => {dash.ctx.triggered_id}")
    raise PreventUpdate
  
  
@dash.callback(
    Output({'type': 'hidden_div_action_str', 'index': MATCH}, 'children'),
    Input({'type': 'action_property_str', 'index': MATCH}, 'value')
)
def on_action_property_str(value):
    """
    Callback for updating float properties
    """
    if not dash.ctx.triggered_id:
        raise PreventUpdate
        
    print(f"on_action_property_str({value}) => {dash.ctx.triggered_id}")
    raise PreventUpdate
