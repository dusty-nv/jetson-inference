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

from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

from server import Server
from datetime import datetime


def create_alerts():
    style={
        'position': 'absolute', 
        'width': '100%', 
        'bottom': 0, 
        'zIndex': 9999
    }
    
    return html.Div([
        dbc.Alert('Placeholder Text', color='#444444', style=style, dismissable=True, is_open=False, id='alerts'),
        dcc.Store(id='alert_count', data=0),
        dcc.Interval(id='alert_timer', interval=1000)
    ])


@dash.callback(Output('alerts', 'children'),
               Output('alerts', 'is_open'),
               Output('alerts', 'duration'),
               Output('alert_count', 'data'),
               Input('alert_timer', 'n_intervals'),
               State('alert_count', 'data'))
def refresh_alerts(n_intervals, alert_count):
    request = Server.request('/status')
    alerts = request.json()['alerts']
    
    if len(alerts) <= alert_count:
        raise PreventUpdate

    children = []
    max_duration = 1
    
    for n in range(alert_count, len(alerts)):
        max_duration = max(max_duration, alerts[n][3]) if (max_duration > 0 and alerts[n][3] > 0) else 0
        text = f"[{datetime.fromtimestamp(alerts[n][2]).strftime('%H:%M:%S')}]  {alerts[n][0]}"
        children.extend([html.Span(text, style={'color': level_to_color(alerts[n][1]), 'fontFamily': 'monospace'}), html.Br()])
        
    return children, True, max_duration, len(alerts)
    
    
def level_to_color(level):
    if level == 'success': return 'limegreen'
    elif level == 'warning': return 'orange'
    elif level == 'error': return 'orange'
    return '#BBBBBB'
