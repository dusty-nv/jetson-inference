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
import plotly.graph_objects as go

from dash import dcc, html, Input, Output
from dash_bootstrap_templates import load_figure_template

from .card import create_card, card_callback

from server import Server
from datetime import datetime


load_figure_template('darkly')


def create_event_timeline():  
    children = [
        dcc.Graph(id='event_timeline_graph'), #, animate=True),
        dcc.Interval(id='event_timeline_timer', interval=500)
    ]
    
    return create_card(
        children,
        title=f"Event Timeline", 
        width=6,
        height=12,
        id='event_timeline'
    )
   
   
@dash.callback(Output('event_timeline_graph', 'figure'),
               Input('event_timeline_timer', 'n_intervals'))
def refresh_timeline(n_intervals):
    request = Server.request('/events')
    records = request.json()
    classes = {}

    for event in records:
        label = event[7]
        
        if label not in classes:
            classes[label] = {'x': [], 'y': []}
            
        for timestamp, score in event[10]:
            classes[label]['x'].append(datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f'))
            classes[label]['y'].append(score * 100)

        classes[label]['x'].append(None)
        classes[label]['y'].append(None)

    fig = go.Figure()
    
    def short_label(label, length=15):
        return f"{label[0:length]}..." if len(label) > length else label
    
    for label, data in classes.items():
        fig.add_trace(go.Scatter(name=short_label(label), x=data['x'], y=data['y'], connectgaps=False))

    fig.update_layout(
        template='darkly',
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        #'uirevision': 0,  # https://community.plotly.com/t/preserving-ui-state-like-zoom-in-dcc-graph-with-uirevision-with-dash/15793
    )
    
    return fig

           
@card_callback(Input('navbar_event_timeline', 'n_clicks'))
def open_timeline(n_clicks):
    if n_clicks > 0:
        return create_event_timeline()  
    else:
        return None