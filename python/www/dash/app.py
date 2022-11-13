#!/usr/bin/env python3
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

from dash import Dash, html, dcc, Input, Output, State, ClientsideFunction
from server import Server

import argparse
import json
import dash_bootstrap_components as dbc

external_scripts = ["https://webrtc.github.io/adapter/adapter-latest.js"]
external_stylesheets = [dbc.themes.DARKLY] #[dbc.themes.SLATE] #[dbc.themes.FLATLY] #[dbc.themes.SUPERHERO] #["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app_title = "WebRTC Dashboard"  # make this configurable
app = Dash(__name__, external_scripts=external_scripts, external_stylesheets=external_stylesheets, title=app_title)

navbar = dbc.NavbarSimple(
    children=[
        #dbc.NavItem(dbc.NavLink("Page 1", href="#")),
        dbc.NavItem(dbc.NavLink("Add Stream", id="navbar_add_stream")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Page 2", href="#"),
                dbc.DropdownMenuItem("Page 3", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand=app_title,
    #brand_href="#",
    color="primary",
    dark=True,
)

add_stream_dialog = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Add Stream")),
        dbc.ModalBody(dbc.Form([
            html.Div([
                dbc.Label("Stream Name", html_for="add_stream_name"),
                dbc.Input(id="add_stream_name", value="/my_stream"), #placeholder="/my_stream"),
                dbc.FormText("Your name of the stream (e.g. location)"),
            ], className="mb-3"),
            html.Div([
                dbc.Label("Video Source", html_for="add_stream_source"),
                dbc.Input(id="add_stream_source", value="/dev/video1"), #placeholder="/dev/video0"),
                dbc.FormText("Camera input: /dev/video*, csi://0, rtsp://, file://"),
            ], className="mb-3"),
        ])),
        dbc.ModalFooter(
            dbc.Button("Create", id="add_stream_submit", className="ms-auto", n_clicks=0)
        )],
    id="add_stream_dialog",
    is_open=False)
        
@app.callback(
    Output("add_stream_dialog", "is_open"),
    Input("navbar_add_stream", "n_clicks"), 
    Input("add_stream_submit", "n_clicks"),
    State("add_stream_dialog", "is_open"),
)
def toggle_add_stream_dialog(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open
    
app.layout = html.Div([
    navbar,
    add_stream_dialog,
    html.Video(id="video_player", controls=True),
    html.Div(id="status_text"),
    dcc.Store(id="stream_config"),
    html.Div(id="hidden_div", style={"display":"none"}),
])

@app.callback(Output("status_text", "children"),
              Output("stream_config", "data"),
              Input("add_stream_submit", "n_clicks"),
              State("add_stream_name", "value"),
              State("add_stream_source", "value"),
              prevent_initial_call=True)
def add_stream(n_clicks, name, source):
    stream = server.streams.add(name, source)
    return [
        html.Div([
            f"Adding stream:  {name}", html.Br(), 
            f"Stream source:  {source}", html.Br(), 
            f"{stream}", html.Br()]),
        json.dumps(stream)]
    

app.clientside_callback(
    ClientsideFunction('webrtc', 'playStream'),
    # specifiy the callback with ClientsideFunction(<namespace>, <function name>)
    # the Output, Input and State are passed in as with a regular callback
    Output('hidden_div', 'children'),
    Input('stream_config', 'data'),
    prevent_initial_call=True
)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8050, type=int)
    parser.add_argument("--rpc-port", default=49565, type=int)
    parser.add_argument("--webrtc-port", default=49567, type=int)
    parser.add_argument("--ssl-cert", default="", type=str)
    parser.add_argument("--ssl-key", default="", type=str)
    #parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()

    # check if HTTPS/SSL requested
    ssl_context = None
    
    if args.ssl_cert and args.ssl_key:
        ssl_context = (args.ssl_cert, args.ssl_key)
        print(f"using SSL cert: {args.ssl_cert}")
        print(f"using SSL key:  {args.ssl_key}")
        
    # start the backend server
    server = Server(host=args.host, rpc_port=args.rpc_port, webrtc_port=args.webrtc_port, ssl_context=ssl_context)
    server = server.start()
    
    # disable code reloading because it starts the app multiple times (https://dash.plotly.com/devtools)
    # https://community.plotly.com/t/dash-multi-page-app-functions-are-called-twice-unintentionally/46450
    # TODO set number of threads??
    app.run_server(host=args.host, port=args.port, ssl_context=ssl_context, debug=True, use_reloader=False)
    
    server.stop()
    print("exiting app")
    