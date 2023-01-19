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
import copy
import dash
import dash_draggable

from dash import dcc, html, Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from .card import card_callbacks, CARD_CONTAINER_PREFIX


def create_grid(children=[], id='grid'):
    """
    Create a grid for draggable/resizable components
    """
    grid = dash_draggable.ResponsiveGridLayout(
        children=children,
        draggableHandle=".card-header",  # the cards can only be moved by dragging the card header
        #clearSavedLayout=True,          # this will reset the layout when the page loads
        #save=False,                     # this will disable client-side saving of the layout
        id=id,
    )
    
    grid_div = html.Div([
        dcc.Store(id=f'{id}_layouts_expanded', data={}),  # stores the original sizes of the cards for expanded/collapsed
        grid
    ])
    
    # https://github.com/MehdiChelh/dash-card_grid/blob/master/src/lib/constants.js
    # the different layout breakpoints in the grid
    LAYOUT_BREAKPOINTS=['lg', 'md', 'sm', 'xs', 'xxs']

    # the default sizes to assign new children added to the grid
    DEFAULT_LAYOUT_SIZES={
        'lg': {'w': 6, 'h': 8},
        'md': {'w': 5, 'h': 7},
        'sm': {'w': 3, 'h': 4},
        'xs': {'w': 4, 'h': 4},
        'xxs': {'w': 2, 'h': 2}
    }

    def find_layout_dict(layouts, breakpoint, index):
        """
        Find the card container's layout dict for a particular layout size from the grid's layouts dict
        
        Parameters:
            layouts (dict)      -- the layouts dict from ResponsiveGridLayout
            breakpoint (string) -- the layout breakpoint, one of LAYOUT_BREAKPOINTS (e.g. 'lg', 'md', 'sm')
            index (int/string)  -- the card index or container ID to lookup
            
        Returns:  the card container's layout dict, or None if it was not found.
                  the card container's layout dict takes the form:
                    {'i': 'card_container_xyz', 'x': 0, 'y': 0, 'w': 4, 'h': 8}
        """
        if not (isinstance(index, str) and index.startswith(CARD_CONTAINER_PREFIX)):
            index = f'{CARD_CONTAINER_PREFIX}{index}'
        
        if breakpoint not in layouts:
            return None
            
        for layout in layouts[breakpoint]:
            if layout['i'] == index:
                return layout
                
        return None
      
    def find_card_style(styles, index):
        """
        Find the style for a particular card index in the list of card styles.
        Once card(s) are removed, the card index no longer corresponds to the actual list index.
        """
        for style in styles:
            if style['cardIndex'] == index:
                return style
        return None
        
    @dash.callback(
        Output(id, 'layouts'),
        Output(id + '_layouts_expanded', 'data'),
        Input(id, 'children'),
        Input({'type': 'card-body', 'index' : ALL}, 'style'),
        State(id, 'layouts'),
        State(id + '_layouts_expanded', "data")
    )
    def sync_layout(children, styles, layouts, layouts_expanded):
        """
        Update the grid's container layouts based on the state of the card components:
        
            * new cards need a default size/position assigned (or restored)
            * cards that have been collapsed need the container minimized
            * cards that have been expanded need the original container size restored
        """
        if isinstance(dash.ctx.triggered_id, dict) and dash.ctx.triggered_id['type'] == 'card-body':
            index = dash.ctx.triggered_id['index']
            style = find_card_style(styles, index)
            
            if not style:
                print(f"sync_layout() couldn't find style for card {index}")
                raise PreventUpdate
            
            if style['visibility'] == 'hidden':
                # the card rolldown is trigged to be collapsed
                # cache it's original size and then shrink the container
                # only update the cache of this particular element, because other cards may be collapsed
                for breakpoint in layouts:
                    layout = find_layout_dict(layouts, breakpoint, index)
                    
                    if layout is None:
                        continue 
                        
                    if index not in layouts_expanded:
                        layouts_expanded[index] = {}
                       
                    if breakpoint not in layouts_expanded[index]:
                        layouts_expanded[index][breakpoint] = {}
                        
                    layouts_expanded[index][breakpoint] = copy.deepcopy(layout)

                    layout['w'] = 2
                    layout['h'] = 1

                return layouts, layouts_expanded
                
            else:
                # the card rolldown is trigged to be expanded
                # restore the original sizes of the card container
                for breakpoint in layouts:
                    layout = find_layout_dict(layouts, breakpoint, index)
                    
                    if layout is None:
                        print(f"showing card, couldn't find layout for breakpoint {breakpoint}  index {index}")
                        continue 
                        
                    layout['w'] = layouts_expanded[str(index)][breakpoint]['w']
                    layout['h'] = layouts_expanded[str(index)][breakpoint]['h']
     
                return layouts, dash.no_update
                
        elif isinstance(dash.ctx.triggered_id, str) and dash.ctx.triggered_id == id:
            # a container was added/removed from the grid
            # check if it doesn't have a layout, and if so assign a default one
            for breakpoint in layouts:
                for child in children:
                    layout = find_layout_dict(layouts, breakpoint, child['props']['id'])
                    style = child['props']['style']
                    
                    if not layout:
                        # this container didn't have a layout for this breakpoint, so create a default one
                        # otherwise if/when the screen resizes to this breakpoint, the container will snap to 1x1
                        layout = {
                            'i': child['props']['id'],
                            'w': style.get('defaultGridWidth', DEFAULT_LAYOUT_SIZES[breakpoint]['w']),
                            'h': style.get('defaultGridHeight', DEFAULT_LAYOUT_SIZES[breakpoint]['h']),
                            'x': 0,
                            'y': 0,
                        }
                        
                        if breakpoint not in layouts:
                            layouts[breakpoint] = []
                            
                        layouts[breakpoint].append(layout)
                    else:
                        # there was an existing layout for this breakpoint, but it was assigned by the grid as 1x1
                        # this happens when a new child gets added to the grid, but only for the active breakpoint
                        if layout['w'] == 1 and layout['h'] == 1:
                            layout['w'] = style.get('defaultGridWidth', DEFAULT_LAYOUT_SIZES[breakpoint]['w'])
                            layout['h'] = style.get('defaultGridHeight', DEFAULT_LAYOUT_SIZES[breakpoint]['h'])

            return layouts, dash.no_update
        
        raise PreventUpdate
        
    #
    # assemble the callback args dynamically from all the card callbacks,
    # so that the manage_cards() callback will be invoked by the components
    # that are used to trigger the creation of the cards
    #
    manage_card_args = [
        Output(id, 'children'),
        Input({'type': 'card-close-button', 'index' : ALL}, 'n_clicks'),
        State(id, "children")
    ]
        
    manage_callback_args = []
    
    for callback in card_callbacks:
        manage_callback_args.extend(callback['args'])

    manage_card_args.extend(manage_callback_args)

    @dash.callback(
        *manage_card_args,
        prevent_initial_call=True
    )
    def manage_cards(close_clicks, children, *args):
        """
        Create or remove a card from the grid depending on if Add or Close buttons were pressed
        It will search for a matching card callback depending on which Input triggered it.
        """
        def get_callback_arg_values(callback):
            callback_arg_values = []
            for callback_arg_signature in callback['args']:
                for i, callback_arg in enumerate(manage_callback_args):
                    if callback_arg == callback_arg_signature:
                        callback_arg_values.append(args[i])
            return callback_arg_values
          
        def create_child(callback):
            new_child = callback['func'](*get_callback_arg_values(callback))
                
            if new_child is None:
                raise PreventUpdate

            for child in children:
                if child['props']['id'] == new_child.id:
                    print(f"ignoring new card created with duplicate ID {new_child.id}")
                    raise PreventUpdate

            return children + [new_child]
                    
        if isinstance(dash.ctx.triggered_id, str):
            for callback in card_callbacks:
                # check if what trigged the callback was from one of the card callback inputs
                if dash.ctx.triggered[0]['prop_id'] not in [str(callback_arg) for callback_arg in callback['args']]:
                    continue

                return create_child(callback)

        elif isinstance(dash.ctx.triggered_id, dict):
            index = dash.ctx.triggered_id['index']
                
            if dash.ctx.triggered_id['type'] == 'card-close-button':
                return [
                    child for child in children 
                    if child["props"]["id"] != f"{CARD_CONTAINER_PREFIX}{index}"
                ]
            else:
                for callback in card_callbacks:
                    # match these types of strings by replacing ["ALL"] with the current index
                    #   callback input:     {"index":["ALL"],"type":"my_button"}.n_clicks
                    #   dash.ctx.triggered:  {"index":0,"type":"my_button"}.n_clicks
                    callback_arg_list = [
                        str(callback_arg).replace('["ALL"]', f"\"{index}\"" if isinstance(index, str) else str(index))  # handle string index names by enclosing them in quotes
                        for callback_arg in callback['args']
                    ]
            
                    if dash.ctx.triggered[0]['prop_id'] not in callback_arg_list:
                        continue
                        
                    return create_child(callback)

        raise PreventUpdate
        
    return grid_div