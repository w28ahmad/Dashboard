import dash_core_components as dcc
import dash_html_components as html

import pandas as pd

# Colors
from cnnDashboard.style import app_colors

layout = html.Div(children=[
    html.Div(children=[
        html.H1(children='Object Detection', 
                style={'margin': '1%', 'font-weight': 'bold', 'color':app_colors['text']}),
    ], style={'backgroundColor': app_colors['background'], 'height':'1200px','width': '87%', 'float':'right'}),
])    