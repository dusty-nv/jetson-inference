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
from config import config as app_config

    
def create_navbar(config=[], id='navbar'):
    """
    Create a navbar components
    """
    return dbc.NavbarSimple(
        children=create_navbar_menus(config, id),
        brand=app_config['dash']['title'],
        #brand_href='#',
        color='primary',
        dark=True,
        id=id,
    )
    
    
def create_navbar_menus(config=[], id='navbar'):
    """
    Create the child components of a navbar
    """
    navbar_items = []
    
    navbar_items += [dbc.NavLink('Test Card', id=f'{id}_test_card', n_clicks=0)]
    navbar_items += [dbc.NavLink('Test Card 2', id=f'{id}_test_card_2', n_clicks=0)]
    
    # populate streams menu
    if len(config) == 0 or len(config['streams']) == 0:   # empty config / default navbar
        navbar_items += [dbc.NavLink('Add Stream', id=f'{id}_add_stream')]
    else:   
        stream_menu_items = [
            dbc.DropdownMenuItem('Add Stream', id=f'{id}_add_stream', n_clicks=0),
            dbc.DropdownMenuItem('Streams', header=True),
        ] 
        
        stream_menu_items += [
            dbc.DropdownMenuItem(name, id={'type': f'{id}_stream', 'index': name}) for name in config['streams']
        ]
        
        navbar_items += dbc.DropdownMenu(
            children=stream_menu_items,
            nav=True,
            in_navbar=True,
            label='Streams',
            id=f'{id}_streams_dropdown',
        )
        
    return navbar_items