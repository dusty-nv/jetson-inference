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

from dash import dcc, html, callback, Input, Output
from config import config

#from layout.test_card import create_test_card

def create_navbar(resources={}):
    """
    Create a navbar component
    """
    return dbc.NavbarSimple(
        children=create_navbar_menus(resources),
        brand=config['dash']['title'],
        #brand_href='#',
        color='primary',
        dark=True,
        id='navbar',
    )
        

def create_navbar_menus(resources={}):
    """
    Create the menu components of a navbar
    """
    navbar = []

    # streams menu 
    stream_menu = [
        dbc.DropdownMenuItem('Add Stream', id='navbar_add_stream', n_clicks=0),
        #dbc.DropdownMenuItem('Streams', header=True),
    ] 
    
    stream_menu += [
        dbc.DropdownMenuItem(name, id={'type': 'navbar_stream', 'index': name}, n_clicks=0) for name in resources.get('streams', [])
    ]
    
    navbar += [
        dbc.DropdownMenu(
            children=stream_menu,
            nav=True,
            in_navbar=True,
            label='Streams',
            id='navbar_streams_dropdown',
    )]
     
    # models menu
    model_menu = [dbc.DropdownMenuItem('Load Model', id='navbar_load_model', n_clicks=0)]
    model_menu += [
        dbc.DropdownMenuItem(name, id={'type': 'navbar_model', 'index': name}, n_clicks=0) for name in resources.get('models', [])
    ]
    
    navbar += [
        dbc.DropdownMenu(
            children=model_menu,
            nav=True,
            in_navbar=True,
            label='Models',
            id='navbar_models_dropdown',
    )]
    
    # events menu
    events_menu = [
        dbc.DropdownMenuItem('Event Table', id='navbar_event_table', n_clicks=0),
        dbc.DropdownMenuItem('Timeline', id='navbar_event_timeline', n_clicks=0),
        dbc.DropdownMenuItem('Actions', id='navbar_actions', n_clicks=0),
    ]
    
    navbar += [
        dbc.DropdownMenu(
            children=events_menu,
            nav=True,
            in_navbar=True,
            label='Events',
            id='navbar_events_dropdown',
    )]

    # config menu
    config_menu = [dbc.DropdownMenuItem('Help', id='navbar_help', href="https://github.com/dusty-nv/jetson-inference/blob/master/docs/webrtc-dash.md", target='_blank')]
    
    navbar += [
        dbc.DropdownMenu(
            children=config_menu,
            nav=True,
            in_navbar=True,
            label='Settings',
            id='navbar_config_dropdown',
    )]
    
    return navbar


@dash.callback(
    Output('navbar', 'children'),
    Input('server_resources', 'data')
)
def refresh_nav(resources):
    """
    Refresh the navbar structure on server updates
    """
    #print("refreshing page navigation")
    return create_navbar_menus(resources)
       