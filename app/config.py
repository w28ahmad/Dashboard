import os, sys
from dotenv import load_dotenv

import dash
from flask import Flask
from flask_login import login_required

BASEDIR = os.path.join(os.getcwd(), "..")
sys.path.append(BASEDIR)

#Load environment Variables
load_dotenv(os.path.join(BASEDIR, '.env'))

SQLALCHEMY_USERNAME=os.getenv("SQLALCHEMY_USERNAME")
SQLALCHEMY_PASSWORD=os.getenv("SQLALCHEMY_PASSWORD")

class BaseConfig:
    SQLALCHEMY_USERNAME
    SQLALCHEMY_PASSWORD
    SECRET_KEY = os.getenv("LOGIN_MANAGER_SECRET_KEY")

# Create app, configure env, register dashapp/extentions/blueprints
def create_app():
    from models.model import User
    
    server = Flask(__name__)
    server.config.from_object(BaseConfig)
    server.config['SQLALCHEMY_DATABASE_URI'] = f'mysql://{SQLALCHEMY_USERNAME}:{SQLALCHEMY_PASSWORD}@localhost/user'
    
    register_dashapps(server)
    register_extensions(server)
    register_blueprints(server)
    
    return server

# Add dash-dashboard routes to main
def register_dashapps(app):
    from dashboard.layout import layout
    from dashboard.callbacks import register_callbacks    

    # Meta tags for viewport responsiveness
    meta_viewport = {"name": "viewport", "content": "width=device-width, initial-scale=1, shrink-to-fit=no"}
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    
    
    dashapp = dash.Dash(__name__,
                         server=app,
                         url_base_pathname='/dashboard/',
                         external_stylesheets=external_stylesheets,
                         meta_tags=[meta_viewport])
    
    # Adding External html
    print(os.path.dirname(os.path.realpath(__file__)))
    dashapp.index_string = open("./app/assets/templates/dashboard.html", "r").read()

    with app.app_context():
        dashapp.title = 'Dashapp'
        dashapp.layout = layout
        register_callbacks(dashapp)
        
    # For dash authentication
    _protect_dashviews(dashapp)

# Add authentication to dashviews
def _protect_dashviews(dashapp):
    for view_func in dashapp.server.view_functions:
        if view_func.startswith(dashapp.config.url_base_pathname):
            dashapp.server.view_functions[view_func] = login_required(dashapp.server.view_functions[view_func])
         
# Add database and login authentication to server
def register_extensions(server):
    from extensions import db
    from models.model import User
    from extensions import login_manager
    from extensions import migrate
    
    db.init_app(server)
    login_manager.init_app(server)
    login_manager.login_view = 'login.login'
    migrate.init_app(server, db)
    
    # Connect users to their cookies
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id)) 

# Register other routes/blueprints
def register_blueprints(server):
    from routes.login import login_api
    from routes.special_routes import corner_case_api

    server.register_blueprint(login_api, url_prefix='/login')
    server.register_blueprint(corner_case_api)