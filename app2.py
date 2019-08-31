import os
import pathlib
import statistics
from collections import OrderedDict

import pathlib as pl
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output, State

import utils

table_header_style = {
    "backgroundColor": "rgb(2,21,70)",
    "color": "white",
    "textAlign": "center",
}


app = dash.Dash(__name__)
server = app.server

APP_PATH = str(pl.Path(__file__).parent.resolve())

app.layout = html.Div(
    className="",a
    children=[
        html.Div(
            className="banner",
            children=[
                html.H2("Gantry Design")
            ]
        ),
        html.Div(
            className="container",
            children=[
                html.Div(
                    className="row",
                    style={},
                    children=[
                        html.Div(
                            className="four columns pkcalc-settings",
                            children=[
                                html.P(["Study Design"]),
                                html.Div(
                                    [
                                        html.Label(
                                            [
                                                html.Div(["Radius"]),
                                                dcc.Input(
                                                    id="radius",
                                                    type="number",
                                                    value=250,
                                                    # debounce=True,
                                                    min=0,
                                                    max=1000
                                                )
                                            ]
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)
                                        # html.Label(
                                        #     [
                                        #         html.Div(["Length1"]),
                                        #         dcc.Input(
                                        #             id="length1",
                                        #             type="number",
                                        #             value=50,
                                        #             # debounce=True,
                                        #             min=0,
                                        #             max=1000
                                        #         ),
                                        #     ]
                                        # ),
                                        # html.Label(
                                        #     [
                                        #         html.Div(["Alpha1"]),
                                        #         dcc.Input(
                                        #             id='alpha1',
                                        #             type='number',
                                        #             value=2,
                                        #             min=0,
                                        #             max=90
                                        #         ),
                                        #     ]
                                        # ),
                                        # html.Label(
                                        #     [
                                        #         html.Div(["Alpha2"]),
                                        #         dcc.Input(
                                        #             id='alpha2',
                                        #             type='number',
                                        #             value=15,
                                        #             min=0,
                                        #             max=90
                                        #         ),
                                        #     ]
                                        # ),
                                        # html.Label(
                                        #     [
                                        #         html.Div(["Head Offset"]),
                                        #         dcc.Input(
                                        #             id='head_offset',
                                        #             type='number',
                                        #             value=3,
                                        #             min=0,
                                        #             max=90
                                        #         ),
                                        #     ]
                                        # ),
                                        # html.Label(
                                        #     [
                                        #         html.Div(["a"]),
                                        #         dcc.Input(
                                        #             id='a',
                                        #             type='number',
                                        #             value=0.4,
                                        #             min=0,
                                        #             max=2
                                        #         ),
                                        #     ]
                                        # ),
                                        # html.Label(
                                        #     [
                                        #         html.Div(["b"]),
                                        #         dcc.Input(
                                        #             id='b',
                                        #             type='number',
                                        #             value=1.3,
                                        #             min=0,
                                        #             max=2
                                        #         ),
                                        #     ]
                                        # ),
                                        # html.Label(
                                        #     [
                                        #         html.Div(["TOB"]),
                                        #         dcc.Input(
                                        #             id='TOB',
                                        #             type='number',
                                        #             value=1.0,
                                        #             min=0,
                                        #             max=2
                                        #         ),
                                        #     ]
                                        # )


@app.callback(
    [Input('radius', 'value')]
    # ,
     # Input('length1', 'value'),
     # Input('alpha1', 'value'),
     # Input('alpha2', 'value'),
     # Input('head_offset', 'value'),
     # Input('a', 'value'),
     # Input('TOB', 'value'),
     # Input('b', 'value')])

if __name__ == "__main__":
    app.run_server(debug=True)
