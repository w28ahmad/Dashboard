import sys, os
import dash
import dash_core_components as dcc
import dash_html_components as html

BASEDIR = os.getcwd()
sys.path.append(BASEDIR)
from utils.load_data import stock_data

start = '2017-04-22'
end = '2018-04-22'
name = "AMZN"
df = stock_data(start, end, name, BASEDIR)
df["Mid-Values"] = (df["High"]+df["Low"])/2
df.reset_index(inplace=True)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Stock Analysis'),

    dcc.Input(
        placeholder='Enter a value...',
        type='text',
        value=''
    ),

    dcc.Graph(
        id='example-graph',
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

if __name__ == '__main__':
    app.run_server(debug=True)