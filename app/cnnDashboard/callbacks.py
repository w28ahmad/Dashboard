from dash.dependencies import Input, Output, State
import dash_html_components as html
import plotly
import plotly.graph_objs as go
from .detection import setup_image_directory, save_as_jpg, image_prediction, read_file
import datetime

# from webapp.app.config import BASEDIR

# Colors
from cnnDashboard.style import app_colors

def register_callbacks(dashapp):
    def parse_contents(contents, filename, date):
        b64String=contents.split('base64,')[1]
        prefix = contents.split('base64,')[0]
        setup_image_directory()
        save_as_jpg(filename, b64String)
        image_prediction()
        new_contents = read_file(filename, prefix)
        return html.Div([
        html.H5(filename),
        html.P(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=new_contents, style={"width": "100%"}),
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