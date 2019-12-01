from dash.dependencies import Input, Output, State
import dash_html_components as html
import plotly
import plotly.graph_objs as go

import datetime

# from webapp.app.config import BASEDIR

# Colors
from cnnDashboard.style import app_colors

def register_callbacks(dashapp):
    def parse_contents(contents, filename, date):
        return html.Div([
        html.P(filename),
        html.P(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents, style={"width": "100%"}),
    ], 
    style={
        'backgroundColor': app_colors['background'],
        'height':'1000px',
        'margin-left': "2%",
        # 'width': '87%',
        'float':'right'
})
    
    @dashapp.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
    def update_output(list_of_contents, list_of_names, list_of_dates):
        if list_of_contents is not None:
            children = [
                parse_contents(list_of_contents, list_of_names, list_of_dates)
                ]
            return children