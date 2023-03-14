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
    
    #print('ACTION_TYPES:')
    #print(action_types)

    children = [
        html.Div([
            dbc.Select( 
                id='create_action_type',
                placeholder="Create new action...",
                options=[{'label': f"{type['class']} ({type['name']})", 'value': type['name']} for type in action_types.values()],
            ),
            dbc.Button('Create', id='create_action_button', className='ms-2', n_clicks=0),
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),
        html.Hr(),
        html.Div(create_action_settings(), id='action_settings')
    ]
    
    return children
   

def create_action_settings( expanded_actions=[] ):
    """
    Create components for configuring each action
    """
    actions = Server.request('/actions').json()
    children = []
    
    #print('ACTIONS:')
    #print(actions)
    
    for action in actions:
        is_expanded = action['id'] in expanded_actions
        
        children += [
            html.Div([
                dbc.Switch(
                    id={'type': 'action_enabled', 'index': action['id']}, 
                    label=f"[{action['id']}] {action['name']}",
                    value=action['enabled'],
                ),
                html.I(
                    id={'type': 'action_expand', 'index': action['id']}, 
                    className=rolldown_class_name(is_expanded), 
                    n_clicks=0,
                ),
                html.Div(id={'type': 'hidden_div_action', 'index': action['id']}, style={'display':'none'})
           ], style={'display': 'flex', 'justifyContent': 'space-between'})
        ]
        
        properties = []
        
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
            elif property['type'] == 'bool':
                control = dbc.Switch(
                    value=property['value'], 
                    id={'type': 'action_property_bool', 'index': index},
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

            properties += [
                dbc.Row([dbc.Col(property_name, width=3), dbc.Col(control)], align='center', className='mb-2'),
                html.Div(id={'type': f"hidden_div_action_{property['type']}", 'index': index}, style={'display':'none'})
            ]
 
        children += [
            dbc.Collapse(
                properties,
                is_open=is_expanded,
                style={'backgroundColor': '#3A3A3A'},
                className='px-2 pt-2 pb-1 border border-dark rounded',
                id={'type': 'action_properties', 'index': action['id']},
            )
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
    
    action = Server.request('POST', f"/actions", json={'type': value}).json()
    return create_action_settings(expanded_actions=[action['id']])
    
    
@dash.callback(
    Output({'type': 'hidden_div_action', 'index': MATCH}, 'children'),
    Input({'type': 'action_enabled', 'index': MATCH}, 'value'),
)
def on_action_enabled(enabled):
    if not dash.ctx.triggered_id:
        raise PreventUpdate
        
    Server.request('PUT', f"/actions/{dash.ctx.triggered_id['index']}", json={'enabled': enabled})
    raise PreventUpdate
    
 
@dash.callback(
    Output({'type': 'action_properties', 'index': MATCH}, 'is_open'),
    Output({'type': 'action_expand', 'index': MATCH}, 'className'),
    Input({'type': 'action_expand', 'index': MATCH}, 'n_clicks'),
    State({'type': 'action_properties', 'index': MATCH}, 'is_open')
)
def on_action_expand(n_clicks, is_open):
    """
    Callback for expanding/collapsing action properties
    """
    if not dash.ctx.triggered_id:
        raise PreventUpdate
        
    is_open = not is_open
    
    return is_open, rolldown_class_name(is_open)
    
    
def rolldown_class_name(is_expanded):
    """
    Return the FontAwesome icon name for the rolldown
    """
    return 'fa fa-chevron-circle-up fa-lg mt-1' if is_expanded else 'fa fa-chevron-circle-down fa-lg mt-1'
    

@dash.callback(
    Output({'type': 'hidden_div_action_bool', 'index': MATCH}, 'children'),
    Input({'type': 'action_property_bool', 'index': MATCH}, 'value')
)
def on_action_property_bool(value):
    """
    Callback for updating boolean properties
    """
    if not dash.ctx.triggered_id:
        raise PreventUpdate
        
    print(f"on_action_property_bool({value}) => {dash.ctx.triggered_id}")
    
    index = dash.ctx.triggered_id['index'].split('.')
    Server.request('PUT', f"/actions/{index[0]}", json={index[1]: bool(value)})
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
    
    index = dash.ctx.triggered_id['index'].split('.')
    Server.request('PUT', f"/actions/{index[0]}", json={index[1]: int(value)})
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
    
    index = dash.ctx.triggered_id['index'].split('.')
    Server.request('PUT', f"/actions/{index[0]}", json={index[1]: float(value)})
    raise PreventUpdate
  
  
@dash.callback(
    Output({'type': 'hidden_div_action_str', 'index': MATCH}, 'children'),
    Input({'type': 'action_property_str', 'index': MATCH}, 'value')
)
def on_action_property_str(value):
    """
    Callback for updating string properties
    """
    if not dash.ctx.triggered_id:
        raise PreventUpdate
        
    print(f"on_action_property_str({value}) => {dash.ctx.triggered_id}")
    
    index = dash.ctx.triggered_id['index'].split('.')
    Server.request('PUT', f"/actions/{index[0]}", json={index[1]: str(value)})
    raise PreventUpdate
