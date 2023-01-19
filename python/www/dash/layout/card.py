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

from dash import dcc, html, callback, Input, Output, State, MATCH
from config import config as app_config


CARD_CONTAINER_PREFIX = "card_container_"  # this gets prepended to the card names to form the ID of the div container in the grid
CARD_CONTAINER_COUNT = 0                   # keeps track of the number of cards for cards that don't have a unique ID


def create_card(children, title=None, id=None, width=None, height=None, 
                close_button=True, minimize_button=True, settings_button=False):
    """
    Creates a card container around a set of child components.
    The card can be dragged, resized, collapse, and closed.

    Parameters:
        children -- component or list of components to be used as the body of the card
        title (str) -- the name used for the card's title bar
        id (str or int) -- the index or name to be used for the card
        width (int) -- the default width (in grid cells) of the card (the default is 6 cells)
        height (int) -- the default height (in grid cells) of the card (the default is 8 cells)
        close_button (bool) -- if true, there will be a close button in the header (default is true)
        minimize_button (bool) -- if true, there will be a collapse/expand button in the header
        settings_button (bool) -- if true, there will be a settings button added to the header
                                  this can also be a string which sets a unique ID type for the button
        
    If ID is unspecified, it will be the Nth card created.
    If title is unspecified, it will be set to the ID.
    """
    global CARD_CONTAINER_COUNT
    
    if id is None:
        id = CARD_CONTAINER_COUNT
        
    if title is None:
        title = id
        
    CARD_CONTAINER_COUNT += 1
    
    # resize card contents to grid container size
    card_style={    
        "width": '100%',
        "height": '100%',
        "display": "flex",  
        "flexDirection": "column",
        "flexGrow": 0,
        "visibility": "visible",  # used to expand/collapse the card body
        "cardIndex": id,         # this is used to track which card this style belongs to in sync_layout()
    }

    if width is not None:
        card_style['defaultGridWidth'] = width
       
    if height is not None:
        card_style['defaultGridHeight'] = height
    
    # build the header children
    header = [html.H5(title, className="d-inline")]
    
    if close_button:
        header += [dbc.Button(className="btn-close float-end", id={"type": "card-close-button", "index": id})]
        
    if minimize_button:
        header += [dbc.Button("__", className="btn-close float-end", style={"background": "transparent"}, id={"type": "card-collapse-button", "index": id})]
    
    if settings_button:
        header += [dbc.Button("⚙", className="btn-close float-end", style={"background": "transparent"}, id={"type": settings_button if isinstance(settings_button, str) else "card-settings-button", "index": id}, n_clicks=0)]

    # return a card container
    return html.Div(
        dbc.Card([
            dbc.CardHeader(header, className="pe-2"),  # make buttons closer to right (padding)
            dbc.CardBody(children, style=card_style, id={"type": "card-body", "index": id}),
        ],
        style=card_style,
        className="mx-auto",  # adding mb-3 adds a black line above the resizing handle
        id={"type": "card", "index": id}
    ),  
    style=card_style,
    id=f"{CARD_CONTAINER_PREFIX}{id}") # ResponsiveGridLayout needs string ID's on its top-level children


@dash.callback(
    Output({"type": "card-body", "index" : MATCH}, "style"),
    Input({"type": "card-collapse-button", "index" : MATCH}, "n_clicks"),
    State({"type": "card-body", "index" : MATCH}, "style"),
    prevent_initial_call=True
)
def collapse_card(n_clicks, style):
    """
    Dash callback used to trigger a card being expanded/collapsed
    It does this by modifying the visibility in the style of the card body
    """
    if style['visibility'] == 'visible':
        style['visibility'] = 'hidden'
    else:
        style['visibility'] = 'visible'

    return style    


# List of card callback functions
card_callbacks = []

def card_callback(*args, **kwargs):
    """
    Decorator used to register card callbacks with the grid.
    These are used to create new cards based on one or more Input triggers.
    These functions should return a card made with the create_card() function.
    
    For example, when 'my_button' gets pressed below, it will create a new card to be added to the grid:
    
        @card_callback(Input('my_button', 'n_clicks'))
        create_my_card(n_clicks):
            return create_card([f"My card body {n_clicks}"], title=f"My card {n_clicks}", id=f"my_card_{n_clicks}")
     
    The reason these are used is because the grid needs to manage it's own children, but dash Outputs can 
    only have one callback. So this card_callback decorator registers your sub-callback with the grid's callback.
    """
    def inner(func):
        global card_callbacks

        card_callbacks.append({
            'func' : func,
            'args' : args,
            'kwargs' : kwargs
        })
    return inner
