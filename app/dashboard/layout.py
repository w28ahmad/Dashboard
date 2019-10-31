import dash_core_components as dcc
import dash_html_components as html

import pandas as pd

# Defaults empty dataFrame for Initial values 
name = ""
df = pd.DataFrame(columns=["Mid-Values", "Date"])

layout = html.Div(children=[
    html.H1(children='Stock Analysis'),
    
    html.Div(children=[
        dcc.Input(
         id='input',
         placeholder='Stock Name',
         type='text',
         value='AMZN',
        ),
        dcc.Input(
            id='start_date',
            placeholder='Start Date',
            type='text',
            value='2017-04-04',
            ),
        dcc.Input(
            id='end_date',
            placeholder='End Date',
            type='text',
            value='2018-04-04',
            ),
        ], style={'display':'inline'}),
    dcc.Graph(
        id='graph',
        figure={
            'data': [
                {'x': df["Date"], 'y': df["Mid-Values"], 'type': 'line', 'name': name},
                # {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'line', 'name': u'Montréal'},
            ],
            'layout': {
                'title': f'Stock Prices for {name}'
            }
        }
    )
])