# # from app import app
# import os, sys
# import pandas as pd
# from flask import Blueprint

# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output

# BASEDIR = os.getcwd()
# sys.path.append(BASEDIR)
# from utils.load_data import stock_data

# dashboard_api = Blueprint('dashboard', __name__)

# # Defaults empty dataFrame for Initial values 
# name = ""
# df = pd.DataFrame(columns=["Mid-Values", "Date"])

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(external_stylesheets=external_stylesheets, url_base_pathname='/dashboard/')
    
# app.layout = html.Div(children=[
#     html.H1(children='Stock Analysis'),
    
#     html.Div(children=[
#         dcc.Input(
#         id='input',
#         placeholder='Stock Name',
#         type='text',
#         value='AMZN',
#         ),
#         dcc.Input(
#             id='start_date',
#             placeholder='Start Date',
#             type='text',
#             value='2017-04-04',
#             ),
#         dcc.Input(
#             id='end_date',
#             placeholder='End Date',
#             type='text',
#             value='2018-04-04',
#             ),
#         ], style={'display':'inline'}),
#     dcc.Graph(
#         id='graph',
#         figure={
#             'data': [
#                 {'x': df["Date"], 'y': df["Mid-Values"], 'type': 'line', 'name': name},
#                 # {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'line', 'name': u'Montréal'},
#             ],
#             'layout': {
#                 'title': f'Stock Prices for {name}'
#             }
#         }
#     )
# ])
        
# @app.callback(
#     Output(component_id="graph", component_property="figure"),
#     [
#         Input(component_id="input", component_property='value'), 
#         Input(component_id="start_date", component_property="value"),
#         Input(component_id="end_date", component_property='value')
#     ]
# )
# # dynamicly updating the graph from the name, start-date and end-date
# def new_df(name, start, end):
#     print(name, start, end)
#     try:
#         df = stock_data(start, end, name, BASEDIR)
#         df["Mid-Values"] = (df["High"]+df["Low"])/2
#         df.reset_index(inplace=True)
#     except:
#         print("Stock Name not found")
#         df =  pd.DataFrame(columns=["Date", "Mid-Values"])
    
#     return {
#             'data':[{'x': df["Date"], 'y': df["Mid-Values"], 'type':'line','name':name},],
#             'layout':{
#                 'title': f'Stock Prices for {name}'
#                 }
#             }
    

# @dashboard_api.route('/')
# def visualize_data():
#     return app.run_server()