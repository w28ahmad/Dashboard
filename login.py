from flask import Blueprint, abort, render_template
# from database import User

login_api = Blueprint('login', __name__, template_folder="templates")


@login_api.route('/', defaults={'page': 'index'})
@login_api.route('/')
def login():
    try:
        return render_template('login.html')
    except:
        abort(404)
