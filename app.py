import sys, os
import pandas as pd
from dotenv import load_dotenv

from flask import Flask, redirect, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_required, logout_user

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

BASEDIR = os.getcwd()
sys.path.append(BASEDIR)
from utils.load_data import stock_data

#Environment Variables
load_dotenv(os.path.join(BASEDIR, '.env'))
SQLALCHEMY_USERNAME=os.getenv("SQLALCHEMY_USERNAME")
SQLALCHEMY_PASSWORD=os.getenv("SQLALCHEMY_PASSWORD")
SECRET_KEY = os.getenv("LOGIN_MANAGER_SECRET_KEY")

# Configuring flask
server = Flask(__name__)
server.config['SQLALCHEMY_DATABASE_URI'] = f'mysql://{SQLALCHEMY_USERNAME}:{SQLALCHEMY_PASSWORD}@localhost/user'
db = SQLAlchemy(server)
bcrypt = Bcrypt(server)

# Configure LoginManager
from database import User    
server.secret_key = SECRET_KEY
login_manager = LoginManager()
login_manager.init_app(server)

# Connect users to their cookies
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


db.create_all()
db.session.commit()

# Blueprints
from login import login_api
# from data_predict import dashboard_api

#Register all the api blueprnts
server.register_blueprint(login_api, url_prefix='/login')
# server.register_blueprint(dashboard_api, url_prefix='/dashboard')

# Login redirect
@server.route('/')
def goto_login():
    return redirect('/login')

# Route to logout
# Remove all the cookies
@server.route('/logout')
@login_required
def signout():
    logout_user
    session.clear() #! BUG-FIX - One of the cookies still remains even after logout_user
    return redirect('/login')


# # Defaults empty dataFrame for Initial values 
name = ""
df = pd.DataFrame(columns=["Mid-Values", "Date"])


# Creating a Dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server, url_base_pathname='/dashboard/')

# Add login_required to the dash page
for view_func in server.view_functions:
    if view_func.startswith('/dashboard/'):
        server.view_functions[view_func] = login_required(server.view_functions[view_func])

app.layout = html.Div(children=[
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

@app.callback(
    Output(component_id="graph", component_property="figure"),
    [
        Input(component_id="input", component_property='value'), 
        Input(component_id="start_date", component_property="value"),
        Input(component_id="end_date", component_property='value')
    ]
)
# dynamicly updating the graph from the name, start-date and end-date
def new_df(name, start, end):
    print(name, start, end)
    try:
        df = stock_data(start, end, name, BASEDIR)
        df["Mid-Values"] = (df["High"]+df["Low"])/2
        df.reset_index(inplace=True)
    except:
        print("Stock Name not found")
        df =  pd.DataFrame(columns=["Date", "Mid-Values"])
    
    return {
            'data':[{'x': df["Date"], 'y': df["Mid-Values"], 'type':'line','name':name},],
            'layout':{
                'title': f'Stock Prices for {name}'
                }
            }

if __name__ == '__main__':
    # server.run(debug=True, host='0.0.0.0')
    app.run_server(debug=True, host='0.0.0.0')
