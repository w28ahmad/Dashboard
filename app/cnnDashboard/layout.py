import dash_core_components as dcc
import dash_html_components as html

import pandas as pd

# Colors
from cnnDashboard.style import app_colors

layout = html.Div(children=[
    html.Div(children=[# Title
        html.H1(children='Object Detection', 
                style={
                    'margin': '1%',
                    'font-weight': 'bold', 
                    'color':app_colors['text']
                    }),
    ]),
    html.Div(children=[# Upload Image section
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '99%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
        html.Div(id='output-image-upload'),
    ], style={}),
],
    style={
        'backgroundColor': app_colors['background'],
        'height':'1200px',
        'width': '87%',
        'float':'right'
        })    