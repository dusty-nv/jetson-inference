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


def create_event_table():
    columns = [
        dict(id='id', name='ID', hideable=True),
        dict(id='1', name='Begin', hideable=True, type='datetime'),
        dict(id='2', name='End', hideable=True, type='datetime'),
        dict(id='3', name='Frames', hideable=True, type='numeric'),
        dict(id='4', name='Stream', hideable=True),
        dict(id='5', name='Model', hideable=True),
        dict(id='6', name='Class', hideable=True, type='numeric'),
        dict(id='7', name='Label', hideable=True),
        dict(id='8', name='Score', hideable=True, type='numeric', format=dash_table.FormatTemplate.percentage(1)),
        dict(id='9', name='Max Score', hideable=True, type='numeric', format=dash_table.FormatTemplate.percentage(1))
    ]
    
    """
    style_header={
        #'backgroundColor': 'rgb(50, 50, 50)',
        #'color': 'white',
        'whiteSpace': 'normal',
        'height': 'auto',
    },
    style_data={
        #'backgroundColor': 'rgb(70, 70, 70)',
        #'color': 'white',
        'whiteSpace': 'normal',
        'height': 'auto',
        'lineHeight': '15px'
    },
    """
            
    children = [
        dash_table.DataTable(
            data=[], 
            columns=columns, 
            id='event_table',
            page_size=10,
            filter_action='native',
            filter_options={'case': 'insensitive', 'placeholder_text': 'filter...'},
            sort_action='native',
            sort_by=[{'column_id': 'id', 'direction': 'desc'}],
            css=[{'selector': '.show-hide', 'rule': 'display: none'}],
            style_table={'overflowX': 'auto'},
            style_data={'font-size': 14},  #'font-family': 'monospace'
        ),
        dcc.Interval(id='event_refresh_timer', interval=500)
    ]
    
    return create_card(
        html.Div(children, className='dbc-row-selectable'),
        title=f"Events", 
        width=6,
        height=12,
        id='events'
    )

@dash.callback(Output('event_table', 'data'),
               Input('event_refresh_timer', 'n_intervals'))
def refresh_events(n_intervals):
    request = Server.request('/events')
    records = request.json()
    
    #date_format = '%Y-%m-%d %H:%M:%S'
    #date_format = '%-I:%M:%S %p'
    date_format = '%H:%M:%S'
    
    def event_to_dict(event):
        d = {
            'id': event[0],
            '1': datetime.fromtimestamp(event[1]).strftime(date_format),
            '2': datetime.fromtimestamp(event[2]).strftime(date_format),
        }
        
        for n in range(3, 10):
            d[str(n)] = event[n]
        
        return d
        
    return [event_to_dict(event) for event in records]
           
@card_callback(Input('navbar_event_table', 'n_clicks'))
def open_events(n_clicks):
    if n_clicks > 0:
        return create_event_table()  
    else:
        return None