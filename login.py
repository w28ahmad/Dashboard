from flask import Blueprint, abort, render_template, request, redirect, url_for
from flask_bcrypt import Bcrypt
from database import User, db
from flask_login import login_user, login_required

bcrypt = Bcrypt()
login_api = Blueprint('login', __name__, template_folder="templates")


# The default webpage for login
@login_api.route('/')
def login():
    try:
        return render_template('login.html')
    except:
        abort(404)

# Quick route to create a user in the db
@login_api.route('/CreateUser/<email>/<password>')
def CreateUser(email, password):
    db.create_all()
    user = User(email=email, password=bcrypt.generate_password_hash(password).decode('utf-8'))
    db.session.add(user)
    db.session.commit()
    return redirect('/')

# Check if the username and password are in the db
# if they are then return true
# this function is ment to be decorator for other function inside the webapp
@login_api.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
    
    user = User.query.filter_by(email=email).first()
    
    if not user or not bcrypt.check_password_hash(user.password, password):
        print('failed')
        return redirect(url_for('login.test'))
    
    login_user(user, remember=False)
    print('passed')
    return redirect(url_for('/dashboard/'))
    
    
# Random test page && Login Failed page
@login_api.route('/test')
def test():
    return 'Login Failed'

    
        
