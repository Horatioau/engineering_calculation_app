import os
import pathlib
import statistics
from collections import OrderedDict

import pathlib as pl
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output, State

import utils
import math
import matplotlib
matplotlib.use('MacOSX')
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.image as mping
import pdb
import matplotlib.transforms as mtransforms
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import dialogs
from helpers import make_dash_table, create_plot
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go


#app = dash.Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

APP_PATH = str(pl.Path(__file__).parent.resolve())

app.layout = html.Div([
dbc.Row([
    dbc.Col(
        html.Div([
            html.Label(
                 [
                    html.Div(["alpha1"]),
                    dcc.Input(
                        id='alpha1',
                        type='number',
                        value=2
                    ),
                ]),
            ]),
        width={"size": 2, "offset": 1}
    ),
    dbc.Col(
        html.Div([
             html.Label(
             [
                html.Div(["length1"]),
                    dcc.Input(
                        id='length1',
                        type='number',
                        value=50
                    ),
                ]),
            ]),
    width=2),
    dbc.Col(
        html.Div([
             html.Label(
             [
                html.Div(["length2"]),
                    dcc.Input(
                        id='length2',
                        type='number',
                        value=25
                    ),
                ]),
            ]),
    width=2),
    dbc.Col(
        html.Div([
             html.Label(
             [
                html.Div(["radius"]),
                dcc.Input(
                    id='radius',
                    type='number',
                    value=250
                ),
            ]),
        ]),
    width=2),
    dbc.Col(
        html.Div([
            html.Label(
                 [
                    html.Div(["alpha2"]),
                    dcc.Input(
                        id='alpha2',
                        type='number',
                        value=15
                    ),
                ])
            ]),
    width=2)
]),
dbc.Row([
    dbc.Col(
        html.Div([
             html.Label(
             [
                html.Div(["a"]),
                dcc.Input(
                    id='a',
                    type='number',
                    value=.4
                ),
            ]),
        ]),
    width={"size": 2, "offset": 1}),
    dbc.Col(
        html.Div([
            html.Label(
            [
                html.Div(["TOB"]),
                dcc.Input(
                    id='TOB',
                    type='number',
                    value=1.0
                ),
            ]),
        ]),
    width=2),
    dbc.Col(
        html.Div([
             html.Label(
             [
                html.Div(["b"]),
                dcc.Input(
                    id='b',
                    type='number',
                    value=1.3
                )
            ])
        ]),
    width=2),
    dbc.Col(
        html.Div([
             html.Label(
             [
                html.Div(["Head Offset"]),
                dcc.Input(
                    id='head_offset',
                    type='number',
                    value=1.3
                )
            ])
        ]),
    width=2),
    dbc.Col(
        html.Div([
             html.Label(
             [
                html.Div(["Take Up"]),
                dcc.Input(
                    id='take_up',
                    type='number',
                    value=0
                )
            ])
        ]),
    width=2)

]),
dbc.Row([
    dbc.Col(
        html.Div([
            html.Button('Calculate', id='button')
        ]),
        width={"size": 6, "offset": 3}
    )
]),
dbc.Row(
        dbc.Col(
            html.Div([
                html.H1('Plots'),
                    dcc.Tabs(id="tabs", value='tab-1', children=[
                        dcc.Tab(label='Figure 1', value='tab-1'),
                        dcc.Tab(label='Figure 2', value='tab-2'),
                        dcc.Tab(label='Figure 3', value='tab-3'),
                        dcc.Tab(label='Figure 4', value='tab-4'),
                        ]),
                html.Div(id='tabscontent')
            ]),width={"size":10, "offset":1}
        )
    )
])


# helper functions#
def rotation(theta,exist_point,origin):

	theta = np.radians(theta)
	vec = exist_point-origin
	r = np.array(( (np.cos(theta), - np.sin(theta)),
               (np.sin(theta),  np.cos(theta)) ))

	new_point = np.dot(vec,r)+origin
	return new_point

class gantry():
    def __init__(self,x2,y2,angle,length,a,b):
        # x1 and y1 are the pivot points of the lower end of the gantry
        #angle +'ve is anti-clockwise'
        self.x2 = x2
        self.y2 = y2
        self.angle = angle
        self.length = length
        self.b = b
        self.a = a

        if self.length ==2:
            self.take_up = True

        #main attachment points#
        self.x1 = x2 - length*math.cos(math.radians(angle))
        self.y1 = y2 - length*math.sin(math.radians(angle))

        #truss#
        self.truss = np.matrix([[i,-j*b] for i in range(length+1) for j in range(2)])
        theta = -math.radians(angle)
        c , s = math.cos(theta) , math.sin(theta)
        R = np.matrix([[c, -s],[s ,c]])
        self.truss_rotated = np.dot(self.truss,R) + np.matrix([self.x1,self.y1])

        #idlers#
        self.idler_coords = np.matrix([[1+(self.length-2)/4*(i),a] for i in range(5)])
        theta = -math.radians(angle)
        c , s = math.cos(theta) , math.sin(theta)
        R = np.matrix([[c, -s],[s ,c]])
        self.idler_coords_rotated = np.dot(self.idler_coords,R)+np.matrix([self.x1,self.y1])


class trestle():
	def __init__(self,x0,y0,y1):
		self.x0 = x0
		self.x1 = x0
		self.y0 = y0
		self.y1 = y1
		self.coordsx = [self.x0,self.x1]
		self.coordsy = [self.y0,self.y1]

class ground_class():
    def __init__(self, xcoords,ycoords):
        self.x = xcoords
        self.y = ycoords
        self.grid = np.matrix([xcoords,ycoords])


class belt_properties():
    def __init__(self,length1,length2,alpha1,alpha2, TOB, radius):
        #calculate key points#
        self.radius = radius
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.TOB = TOB
        self.length1 = length1
        self.length2 = length2

        self.TP1_x = length1*math.cos(math.radians(alpha1))
        self.TP1_y = length1*math.sin(math.radians(alpha1))+TOB
        self.CIR_x = self.TP1_x - radius*math.sin(math.radians(alpha1))
        self.CIR_y = self.TP1_y + radius*math.cos(math.radians(alpha1))
        self.TP2 = rotation((alpha1-alpha2),np.matrix([self.TP1_x,self.TP1_y]),np.matrix([self.CIR_x,self.CIR_y]))
        self.TP2_x = self.TP2[0,0]
        self.TP2_y = self.TP2[0,1]
        self.Head_x = self.TP2_x+length2*math.cos(math.radians(alpha2))
        self.Head_y = self.TP2_y+length2*math.sin(math.radians(alpha2))


def gradient(x, bp):
	if x > bp.Head_x:
		return 0
	else:
		return (belt(x, bp)-belt(x-0.1, bp))/(.1)

def ground_interp(x, ground):
	# a function to compute the y value of the ground for the given x
	return np.interp(x,ground.x,ground.y)

def belt(x, bp):
	# a function to computer the y value of the belt at the given x
	if x < bp.TP1_x:
		return x*math.tan(math.radians(bp.alpha1))+bp.TOB
	elif x < bp.TP2_x:
		return bp.CIR_y-math.sqrt(bp.radius**2-(x-bp.CIR_x)**2)
	elif x < bp.Head_x and x > bp.TP2_x:
		xcoords = [bp.TP2_x,bp.Head_x]
		ycoords = [bp.TP2_y,bp.Head_y]
		return np.interp(x,xcoords,ycoords)

def dist(x1,y1,x2,y2):
	return math.sqrt((y2-y1)**2+(x2-x1)**2)

def get_pack_heights(gan_ob, bp): # OOP method !! gan_ob is a class
	idler_pack = []
	for coords in gan_ob.idler_coords_rotated: #coords is a matrix list of x&y coords of the gantry nodes in position
		pack_height = belt(coords[0,0], bp)-coords[0,1]
		idler_pack.append(pack_height)
	min_pack_h2 = min(idler_pack)
	max_pack_h2 = max(idler_pack)
	return min_pack_h2, max_pack_h2

def plot_belt(bp):

    xcircle = np.linspace(bp.TP1_x,bp.TP2_x,50)-bp.CIR_x
    ycircle = -np.sqrt(bp.radius**2-xcircle**2)+bp.CIR_y

    x_belt = [bp.TP1_x]
    for i in xcircle:
    	x_belt.append(i+bp.CIR_x)
    x_belt.append(bp.Head_x)

    y_belt = [bp.TP1_y]
    for i in ycircle:
    	y_belt.append(i)
    y_belt.append(bp.Head_y)

    xcircle = np.linspace(bp.TP1_x,bp.TP2_x,50)-bp.CIR_x
    ycircle = -np.sqrt(bp.radius**2-xcircle**2)+bp.CIR_y


    data = {"xb": x_belt, "yb": y_belt}
    data2 = {"xc": xcircle+bp.CIR_x, "yc": ycircle}
    df = pd.DataFrame(data)
    df2 = pd.DataFrame(data2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["xb"], y=df["yb"], mode='lines+markers', name= 'Belt'))
    fig.add_trace(go.Scatter(x=df2["xc"], y=df2["yc"], mode='lines+markers', name= 'Circle'))
    fig.add_trace(go.Scatter(x=[0,bp.TP1_x], y=[bp.TOB,bp.TP1_y], mode='lines+markers', name= 'Take up?'))

    fig.update_xaxes(range=[0, bp.Head_x])
    fig.update_yaxes(range=[0, bp.Head_y*2])

    fig.update_layout(
        title=go.layout.Title(
            text="plot_belt",
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="x Axis"
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="y Axis",
            )
        )
    )

    return fig


def plot_ground(ground):
    data = {"x": ground.x, "y": ground.y}
    df = pd.DataFrame(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["x"], y=df["y"], mode="lines", name='Ground', line=dict(color ='darkred', width =1.5)))

    fig.update_layout(
        title=go.layout.Title(
            text="plot_ground",
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="x Axis"
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="y Axis",
            )
        )
    )

    return fig

def plot_truss(gantry_list):

    for j in range(len(gantry_list)):
    	xtruss = [i[0,0] for i in gantry_list[j].truss_rotated]
    	xtruss.append(gantry_list[j].truss_rotated[1,0])
    	ytruss = [i[0,1] for i in gantry_list[j].truss_rotated]
    	ytruss.append(gantry_list[j].truss_rotated[1,1])


    data = {"x_truss": xtruss,
            "y_truss": ytruss
            }

    df = pd.DataFrame(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["x_truss"], y=df["y_truss"], mode="lines+markers",line=dict(color='black', width = .5), name='Truss Plot'))

    fig.update_layout(
        title=go.layout.Title(
            text="plot_truss",
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="x Axis"
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="y Axis",
            )
        )
    )

    return fig

def plot_steel(llm_list, ebm_list, gantry_list, trestle_list):

    ### steel line plotting
    x_llm = [column[0] for column in llm_list]
    y_llm = [column[1] for column in llm_list]
    x_ebm = [column[0] for column in ebm_list]
    y_ebm = [column[1] for column in ebm_list]
    x_gan = [i.x1 for i in gantry_list]
    x_gan.insert(0,gantry_list[0].x2)
    y_gan = [i.y1 for i in gantry_list]
    y_gan.insert(0,gantry_list[0].y2)

    x_tres=[i.coordsx for i in trestle_list]
    y_tres=[i.coordsy for i in trestle_list]

    x_gan_anno = [i.x1+i.length/2 for i in gantry_list]
    y_gan_anno = [i.y1 for i in gantry_list]
    text_gan_anno = [i.length for i in gantry_list]

    text_tres_anno = [int(round((i.y1-i.y0),0)) for i in trestle_list]
    x_tres_anno =[i.x0 for i in trestle_list]
    y_tres_anno =[i.y1*.66 for i in trestle_list]


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_llm, y=y_llm, mode='lines+markers', line=dict(color='lightblue', width=1), name="LLM"))
    fig.add_trace(go.Scatter(x=x_ebm, y=y_ebm, mode='lines+markers', line=dict(color='lightgreen', width=1), name="EBM"))
    fig.add_trace(go.Scatter(x=x_gan, y=y_gan, mode='lines',line=dict(color='black', width=0.7), name="gan"))
    fig.add_trace(go.Scatter(x=x_tres, y=y_tres, mode='markers', marker=dict(color='navy', size=10), name="tres"))
    fig.add_trace(go.Scatter(x=x_gan_anno, y=y_gan_anno, text = text_gan_anno,  mode = 'text', textfont=dict(size=12, color = 'red'), name="Gantry Length"))
    fig.add_trace(go.Scatter(x=x_tres_anno, y=y_tres_anno, text = text_tres_anno, mode = 'text', textfont=dict(size=12, color = 'green'), name="Trestle Height"))

    fig.update_layout(
        title=go.layout.Title(
            text="plot_steel",
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="x Axis"
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="y Axis",
            )
        )
    )
    return fig

def procedure(alpha1, length1, length2, radius, alpha2, head_offset, a, TOB,b, take_up):
    # set lists #
    gantry_list=[]
    trestle_list=[]
    llm_list = []
    ebm_list=[]

    #ground points, convert to inputs
    gx = [0,50,51,150,200,400]
    gy = [0,0,0,0,0,0]
    ground = ground_class(gx, gy)

    bp=belt_properties(length1,length2,alpha1,alpha2, TOB, radius)

    #set initial calc parameters#
    x1 = bp.Head_x
    y1 = bp.Head_y

    # start at head end and move back to offset connection#
    angle = degrees(math.atan(gradient(bp.Head_x-head_offset,bp)))
    x_belt = bp.Head_x-head_offset*cos(radians(angle))
    y_belt = bp.Head_y-head_offset*sin(radians(angle))

    #get pivot point of gantry position#
    x2_gantry = x_belt+a*sin(radians(angle))
    y2_gantry = y_belt-a*cos(radians(angle))

    #gantries#
    while (y1 - ground_interp(x1, ground)) > 2.5:

    	#position gantry and check#
    	length = 24
    	angle = degrees(math.atan(gradient(x2_gantry, bp)))

    	min_pack = 100
    	max_pack = 100

    	#add take up tower#
    	#DEBUG to add take up tower at least 1 gantry away from first pivot#
    	if y2_gantry-ground_interp(x2_gantry, ground) < 12 and take_up == 0:
    		length = 2
    		take_up = 1
    		min_pack = 0

    	while min_pack > 0.05:
    		min_pack, max_pack = get_pack_heights(gantry(x2_gantry,y2_gantry,angle,length,a,b), bp)
    		if min_pack <= 0.05:
    			break
    		angle -= 0.1

    	while max_pack > .4:
    		min_pack, max_pack = get_pack_heights(gantry(x2_gantry,y2_gantry,angle,length,a,b), bp)
    		if max_pack <= 0.4:
    			break
    		length -= 6

    	#get parameters#
    	x1 = gantry(x2_gantry,y2_gantry,angle,length,a,b).x1
    	y1 = gantry(x2_gantry,y2_gantry,angle,length,a,b).y1

    	while y1 - ground_interp(x1, ground) < 2.5:
    		length -= 6
    		x1 = gantry(x2_gantry,y2_gantry,angle,length,a,b).x1
    		y1 = gantry(x2_gantry,y2_gantry,angle,length,a,b).y1
    		#pdb.set_trace()
    		if length ==6:
    			break
    		if y1 - ground_interp(x1, ground) >= 2.5:
    			length += 6
    			break

    	#add gantries#
    	gantry_list.append(gantry(x2_gantry,y2_gantry,angle,length,a,b))

    	#add trestles #
    	trestle_list.append(trestle(x1,ground_interp(x1, ground),y1-b)) #check

    	# increment to next gantry#
    	x2_gantry = x1
    	y2_gantry = y1

    # add elevated beam module#
    ebm_list.append([x1,y1])
    while (y1 - ground_interp(x1, ground)) > 1.5:
    	x1 -= 0.1
    	y1 = belt(x1,bp)
    ebm_list.append([x1,y1])

    # add ground module to the end#
    llm_list.append([x1,y1])
    llm_list.append([0,TOB])


    fig1 = plot_steel(llm_list, ebm_list, gantry_list, trestle_list)
    fig2 = plot_truss(gantry_list)
    fig3 = plot_belt(bp)
    fig4 = plot_ground(ground)


    set_lists = [gantry_list,
                trestle_list,
                llm_list,
                ebm_list]

    fig_list = [fig1,
                fig2,
                fig3,
                fig4,
                set_lists]

    return(fig_list)


@app.callback(
    Output('tabscontent', 'children'),
    [Input('button', 'n_clicks'),
     Input('alpha1', 'value'),
     Input('length1', 'value'),
     Input('length2', 'value'),
     Input('radius', 'value'),
     Input('alpha2', 'value'),
     Input('head_offset', 'value'),
     Input('a', 'value'),
     Input('TOB', 'value'),
     Input('b', 'value'),
     Input('take_up', 'value'),
     Input('tabs', 'value')]
)
def update(button, alpha1, length1, length2, radius, alpha2,head_offset,a, TOB,b, take_up, tabs):
    if button is None:
        raise PreventUpdate
    calculations = procedure(alpha1, length1, length2, radius, alpha2,head_offset,a,TOB,b, take_up)

    if tabs == 'tab-1':
        return html.Div([
                dcc.Graph(id='graph-1-tabs',figure=calculations[0])
            ])
    elif tabs == 'tab-2':
        return html.Div([
                dcc.Graph(id='graph-2-tabs',figure=calculations[1])
            ])
    elif tabs == 'tab-3':
        return html.Div([
                dcc.Graph(id='graph-3-tabs',figure=calculations[2])
            ])
    elif tabs == 'tab-4':
        return html.Div([
                dcc.Graph(id='graph-4-tabs',figure=calculations[3])
            ])



if __name__ == '__main__':
    app.run_server(debug=True)
