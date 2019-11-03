import dash_core_components as dcc
import dash_html_components as html

import pandas as pd

layout = html.Div(children=[
    html.Div(children=[
        html.H1(children='Stock Analysis', style={'margin': '1%', 'font-weight': 'bold'}),
        
        html.Div(children=[
            html.Div(children=[ # inputs
                html.P('Stock Name:'),
                dcc.Input(
                id='input',
                placeholder='Stock Name',
                type='text',
                value='AMZN',
                )
            ], style={'margin-top':'2%'}),
            html.Div(children=[# inputs
                html.P('Start Date:'),
                dcc.Input(
                id='start_date',
                placeholder='Start Date',
                type='text',
                value='2017-04-04',
                )
            ], style={'margin-top':'2%'}),
            html.Div(children=[# inputs
                html.P('End Date:'),
                dcc.Input(
                id='end_date',
                placeholder='End Date',
                type='text',
                value='2018-04-04',
                )
            ],  style={'margin-top':'2%'}),
            html.Div(children=[# inputs
                html.P('Prediction Amount:'),
                dcc.Input(
                id='predict_amount',
                placeholder='',
                type='text',
                value='',
                )
            ],  style={'margin-top':'2%'}),
            html.Div([
                dcc.Checklist(# checkbox
                    id='predict',
                    options=[
                        {'label': 'Predict Future', 'value': 'predict'},
                    ],
                    value=[]
                )
            ], style={'margin-top':'5%'}),
        ], style={'margin':'2%', 'display':'inherit'}),
        html.Div([
                dcc.Graph(
                id='graph-prediction',
                )        
        ], style={
                'width':'80%', 
                'float':'right',
                'display':'inherit'
                })

    ], style={'display':'inline-block', 'width':'100%'}),
    html.Div(children=[# Sentiment analysis section
        html.H1(children='Stock Sentiment Analysis', style={'font-weight': 'bold', 'display':'inherit'})
    ], className='title', style={'display':'inline-block',
                                 'margin':'1%'}
    ),
    
    html.Div(children=[
        dcc.Graph(
            id='graph-sentiment',
        )
    ])
    
])