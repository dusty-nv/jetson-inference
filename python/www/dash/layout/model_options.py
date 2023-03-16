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

import os
import dash
import dash_bootstrap_components as dbc

from dash import dcc, html, callback, Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from server import Server


def create_model_dialog(model={}):
    """
    Create the top-level dialog container used for creating/configuring models.
    It's children will be created dynamically in create_model_options() below.
    """
    return dbc.Modal(create_model_options(model), id='model_options_dialog', is_open=False)
    
    
def create_model_options(model={}):
    """
    Create the dialog body used for creating/configuring models.
    """
    children = [dbc.ModalHeader(dbc.ModalTitle(model.get('name', 'Load Model')))]
 
    tabs = dbc.Tabs([
            dbc.Tab(label='Pre-trained', tab_id='model_tab_pretrained'),
            #dbc.Tab(label='Train', tab_id='model_tab_train'),
            dbc.Tab(label='Import', tab_id='model_tab_import'),
        ],
        id='model_tabs',
        active_tab='model_tab_pretrained',
    )

    children += [dbc.ModalBody([tabs, html.Div(id='model_content', className='pt-2')])]
    children += [dbc.ModalFooter(dbc.Button('Load', id={'type': 'model_options_submit', 'index': 0}, className='ms-auto', n_clicks=0))]
    
    return children
  
    
def create_pretrained_options():
    """
    Create the form for the pre-trained models tab
    """
    return dbc.Form([
        html.Div([
            dbc.Label('Model Type', html_for='model_pretrained_type'),
            dbc.Select( 
                options=[
                    {'label': 'Classification', 'value': 'classification'},
                    {'label': 'Object Detection', 'value': 'detection'},
                    #{'label': 'Semantic Segmentation', 'value': 'segmentation'},
                    #{'label': 'Pose Estimation', 'value': 'pose'},
                ],
                value='classification',
                id='model_pretrained_type',
            ),
            dbc.FormText("The type of model (e.g. image classification, object detection, ect.)"),
        ], className='mb-3'),
        
        html.Div([
            dbc.Label('Network', html_for='model_pretrained_network'),
            dbc.Select(id='model_pretrained_network', value='googlenet'),
            dbc.FormText("The model network architecture to load"),
        ], className='mb-3'),
        
        #html.Div(id='model_pretrained_content', className='mb-3'),
        html.Div(id='hidden_div_model_pretrained', style={'display':'none'}),
    ])


@dash.callback(
    Output('model_pretrained_network', 'options'),
    Output('model_pretrained_network', 'value'),
    Input('model_pretrained_type', 'value')
)
def list_pretrained_models(type):
    """
    Return a drop-down list of pre-trained model options that can be selected
    """
    if type == 'classification':
        return [
            {'label': 'Alexnet', 'value': 'alexnet'},
            {'label': 'Googlenet', 'value': 'googlenet'},
            {'label': 'ResNet-18', 'value': 'resnet-18'},
            {'label': 'ResNet-50', 'value': 'resnet-50'},
        ], 'googlenet'
    elif type == 'detection':
        return [
            {'label': 'SSD-Mobilenet-v1', 'value': 'ssd-mobilenet-v1'},
            {'label': 'SSD-Mobilenet-v2', 'value': 'ssd-mobilenet-v2'},
            {'label': 'SSD-Inception-v2', 'value': 'ssd-inception-v2'},
        ], 'ssd-mobilenet-v2'
    else:
        return [], None

   
def create_import_options():
    """
    Create the form for the import model tab
    """
    return dbc.Form([
        html.Div([
            dbc.Label('Model Type', html_for='model_import_type'),
            dbc.Select( 
                options=[
                    {'label': 'Classification', 'value': 'classification'},
                    {'label': 'Object Detection', 'value': 'detection'},
                ],
                value='classification',
                id='model_import_type',
            ),
            dbc.FormText("The type of model (e.g. image classification, object detection, ect.)"),
        ], className='mb-3'),
        
        html.Div([
            dbc.Label('Model Path', html_for='model_import_path'),
            dbc.Input(id='model_import_path', invalid=True),
            dbc.FormText("Path on the server to the ONNX model to load"),
            dbc.FormFeedback("Provide a path to a valid file on the server", type='invalid'),
        ], className='mb-3'),
        
        html.Div([
            dbc.Label('Labels Path', html_for='model_import_labels'),
            dbc.Input(id='model_import_labels'),
            dbc.FormText("Path on the server to the model's labels.txt file"),
            dbc.FormFeedback("Provide a path to a valid file on the server", type='invalid'),
        ], className='mb-3'),
        
        html.Div([
            dbc.Label('Input Layer', html_for='model_import_layer_input'),
            dbc.Input(id='model_import_layer_input', value='input_0'),
            dbc.FormText("Name of the model's input layer"),
        ], className='mb-3'),
        
        html.Div(id='model_import_content', className='mb-3'),
    ])


@dash.callback(
    Output('model_import_content', 'children'),
    Input('model_import_type', 'value'),
)
def create_import_sub_options(type):
    """
    Create form elements that are specific to the type of model being imported
    """
    if type == 'classification':
        return [
            html.Div([
                dbc.Label('Output Layer', html_for='model_import_classification_layer_output'),
                dbc.Input(id='model_import_classification_layer_output', value='output_0'),
                dbc.FormText("Name of the model's output layer"),
            ], className='mb-3'),
            
            html.Div(id='hidden_div_model_import_classification', style={'display':'none'}),
        ]
    elif type == 'detection':
        return [
            html.Div([
                dbc.Label('Output Layer (Scores)', html_for='model_import_detection_layer_scores'),
                dbc.Input(id='model_import_detection_layer_scores', value='scores'),
                dbc.FormText("Name of the model's scores/coverage output layer"),
            ], className='mb-3'),
            
            html.Div([
                dbc.Label('Output Layer (Bounding Boxes)', html_for='model_import_detection_layer_bbox'),
                dbc.Input(id='model_import_detection_layer_bbox', value='boxes'),
                dbc.FormText("Name of the model's bounding boxes output layer"),
            ], className='mb-3'),
            
            html.Div(id='hidden_div_model_import_detection', style={'display':'none'}),
        ]
    else:
        return []
        
        
@dash.callback(
    Output('model_import_path', 'valid'),
    Output('model_import_path', 'invalid'),
    Input('model_import_path', 'value')
)
def validate_model_import_path(path):
    """
    Validate that the model path exists
    """
    if path and os.path.isfile(path):
        return True, False
    else:
        return False, True
   

@dash.callback(
    Output('model_import_labels', 'valid'),
    Output('model_import_labels', 'invalid'),
    Input('model_import_labels', 'value')
)
def validate_model_import_labels(path):
    """
    Validate that the model labels path exists
    """
    if not path:
        return None, None
    elif os.path.isfile(path):
        return True, False
    else:
        return False, True
        

def model_name_from_path(path):
    """
    Return the directory of a model with it's filename
    '/path/my_model/net.onnx' will return 'my_model/net.onnx'
    """
    return os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
    

@dash.callback(
    Output('hidden_div_model_pretrained', 'children'),
    Input({'type': 'model_options_submit', 'index': ALL}, 'n_clicks'),
    State('model_pretrained_type', 'value'),
    State('model_pretrained_network', 'value'),
)
def model_submit_pretrained(n_clicks, type, network):
    """
    Callback for when the pretrained model form is submitted
    """
    if len(n_clicks) == 0 or n_clicks[0] == 0:
        raise PreventUpdate
        
    print(f"model_submit_pretrained({n_clicks}, {type}, {network})")
    Server.request('POST', 'models', json={'name': network, 'type': type, 'model': network})
    raise PreventUpdate

    
@dash.callback(
    Output('hidden_div_model_import_classification', 'children'),
    Input({'type': 'model_options_submit', 'index': ALL}, 'n_clicks'),
    State('model_import_type', 'value'),
    State('model_import_path', 'value'),
    State('model_import_labels', 'value'),
    State('model_import_layer_input', 'value'),
    State('model_import_classification_layer_output', 'value')
)
def model_submit_import_classification(n_clicks, type, path, labels, layer_input, layer_output):
    """
    Callback for when the import classification model form is submitted
    """
    if len(n_clicks) == 0 or n_clicks[0] == 0:
        raise PreventUpdate
        
    print(f"model_submit_import_classification({n_clicks}, {type}, {path}, {labels}, {layer_input}, {layer_output})")
    
    args = {
        'name': model_name_from_path(path),
        'type': type,
        'model': path,
        'labels': labels,
        'input_layers': layer_input,
        'output_layers': layer_output
    }
    
    Server.request('POST', 'models', json=args)
    raise PreventUpdate


@dash.callback(
    Output('hidden_div_model_import_detection', 'children'),
    Input({'type': 'model_options_submit', 'index': ALL}, 'n_clicks'),
    State('model_import_type', 'value'),
    State('model_import_path', 'value'),
    State('model_import_labels', 'value'),
    State('model_import_layer_input', 'value'),
    State('model_import_detection_layer_scores', 'value'),
    State('model_import_detection_layer_bbox', 'value')
)
def model_submit_import_detection(n_clicks, type, path, labels, layer_input, layer_scores, layer_bbox):
    """
    Callback for when the import detection model form is submitted
    """
    if len(n_clicks) == 0 or n_clicks[0] == 0:
        raise PreventUpdate
        
    print(f"model_submit_import_detection({n_clicks}, {type}, {path}, {labels}, {layer_input}, {layer_scores}, {layer_bbox})")
    
    args = {
        'name': model_name_from_path(path),
        'type': type,
        'model': path,
        'labels': labels,
        'input_layers': layer_input,
        'output_layers': {'scores': layer_scores, 'bbox': layer_bbox}
    }
    
    Server.request('POST', 'models', json=args)
    raise PreventUpdate


@dash.callback(
    Output('model_content', 'children'), 
    Input('model_tabs', 'active_tab')
)
def switch_model_tab(at):
    """
    Switch the content of the current tab
    """
    if at == 'model_tab_pretrained':
        return create_pretrained_options()
    elif at == 'model_tab_train':
        return 'TODO'
    elif at == 'model_tab_import':
        return create_import_options()
        
    raise PreventUpdate
    
    
@dash.callback(
    Output('model_options_dialog', 'is_open'),
    Output('model_options_dialog', 'children'),
    Input('navbar_load_model', 'n_clicks'), 
    Input({'type': 'model_options_submit', 'index': ALL}, 'n_clicks'),
    Input({'type': 'navbar_model', 'index': ALL}, 'n_clicks'),
    State('model_options_dialog', 'is_open'),
)
def show_model_dialog(n1, n2, n3, is_open):
    """
    Callback for triggering to show/hide the model options dialog
    """
    model = {}
    
    #print(f'show_model_dialog({n1}, {n2}, {n3}, {is_open})')
    #print(dash.ctx.triggered_id)

    if not dash.ctx.triggered[0]['value']:
        raise PreventUpdate

    if isinstance(dash.ctx.triggered_id, dict) and dash.ctx.triggered_id['type'] == 'navbar_model':
        model = Server.request(f"models/{dash.ctx.triggered_id['index']}").json()
        
    if is_open:
        return False, dash.no_update
    else:
        return True, create_model_options(model)
   


