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

app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Input(
                id='alpha1',
                type='number',
                value=2
            ),
            dcc.Input(
                id='length1',
                type='number',
                value=50
            ),
            dcc.Input(
                id='radius',
                type='number',
                value=250
            ),
            dcc.Input(
                id='alpha2',
                type='number',
                value=15
            ),
            dcc.Input(
                id='a',
                type='number',
                value=.4
            ),
            dcc.Input(
                id='TOB',
                type='number',
                value=1.0
            ),
            dcc.Input(
                id='b',
                type='number',
                value=1.3
            )
        ])
    ])#,
    #dcc.Graph(id='indicator-graphic')
])


# groundx = [0,50,51,150,200,400]
# groundy = [0,0,0,0,0,0]
# ground = np.matrix([groundx,groundy])


@app.callback(
    #Output('indicator-graphic', 'figure'),
    [Input('alpha1', 'value'),
     Input('length1', 'value'),
     Input('radius', 'value'),
     Input('alpha2', 'value'),
     Input('head_offset', 'value'),
     Input('a', 'value'),
     Input('TOB', 'value'),
     Input('b', 'value')
     ])


if __name__ == '__main__':
    app.run_server(debug=True)
