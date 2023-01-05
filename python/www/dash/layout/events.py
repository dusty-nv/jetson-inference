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

from dash import dcc, html, dash_table, Input, Output
from .card import create_card, card_callback

from server import Server
from datetime import datetime


def create_events():
    columns = [
        {'id': 'id', 'name': 'ID', 'hideable': True},
        {'id': '0', 'name': 'Begin', 'hideable': True, 'type': 'datetime'},
        {'id': '1', 'name': 'End', 'hideable': True, 'type': 'datetime'},
        {'id': '2', 'name': 'Stream', 'hideable': True},
        {'id': '3', 'name': 'Model', 'hideable': True},
        {'id': '4', 'name': 'Class', 'hideable': True, 'type': 'numeric'},
        {'id': '5', 'name': 'Label', 'hideable': True},
        {'id': '6', 'name': 'Score', 'hideable': True, 'type': 'numeric', 'format': dash_table.FormatTemplate.percentage(1)},
        {'id': '7', 'name': 'Max Score', 'hideable': True, 'type': 'numeric', 'format': dash_table.FormatTemplate.percentage(1)}
    ]
    
    children = [
        dash_table.DataTable(
            data=[], 
            columns=columns, 
            id='events_table',
            sort_action='native',
            sort_by=[{'column_id': 'id', 'direction': 'desc'}],
            page_size=10,
            style_header={
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_data={
                'backgroundColor': 'rgb(70, 70, 70)',
                'color': 'white',
                'whiteSpace': 'normal',
                'height': 'auto',
                'lineHeight': '15px'
            },
        ),
        dcc.Interval(id='events_refresh_timer', interval=500)
    ]
    
    return create_card(
        children,
        title=f"Events", 
        width=6,
        height=15,
        id='events'
    )

@dash.callback(Output('events_table', 'data'),
               Input('events_refresh_timer', 'n_intervals'))
def refresh_events(n_intervals):
    request = Server.request('/events')
    records = request.json()
    
    def event_to_dict(event, id):
        d = { str(n) : value for n, value in enumerate(event) }
        d['id'] = id
        d['0'] = datetime.fromtimestamp(d['0']).strftime('%Y-%m-%d %H:%M:%S')
        d['1'] = datetime.fromtimestamp(d['1']).strftime('%Y-%m-%d %H:%M:%S')
        return d
        
    return [
        event_to_dict(event, id)
        for id, event in enumerate(records)
    ]
           
@card_callback(Input('navbar_events', 'n_clicks'))
def open_events(n_clicks):
    if n_clicks > 0:
        return create_events()  
    else:
        return None